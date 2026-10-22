"""Isolated structure-head SFT and DPO experiments on existing BIRD data."""
import argparse,math,os,random,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from structure_dpo_data import ROOT,OUT,SOURCE,EVAL,CONFIG,prepare
from pilot import read,save,collate
import pilot
from experiment import records,append,export_records
from hardpair_batch2 import batches,generate_batch

GROUPS=['structure','sft-cont','dpo']

def balanced_bce(logits,labels):
    if not len(logits):return logits.sum()*0
    losses=F.binary_cross_entropy_with_logits(logits.float(),labels.float(),reduction='none')
    terms=[losses[mask].mean() for mask in [labels>0,labels==0] if bool(mask.any())]
    return torch.stack(terms).mean()

def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta=.1):
    margin=beta*((chosen-rejected)-(reference_chosen-reference_rejected))
    return -F.logsigmoid(margin),margin

def completion_logp(logits,targets):
    return -F.cross_entropy(logits.float(),targets,reduction='sum')

class StructureHeads(nn.Module):
    def __init__(self,hidden,width=128):
        super().__init__();self.width=width
        self.query=nn.ModuleList([nn.Linear(hidden,width) for _ in range(3)])
        self.keys=nn.ModuleList([nn.Linear(hidden,width),nn.Linear(hidden,width),nn.Linear(2*hidden,width)])
        self.bias=nn.Parameter(torch.zeros(3))
    def forward(self,context,tables,columns,edges):
        if len(edges):
            a=columns[edges[:,0]];b=columns[edges[:,1]];features=torch.cat([a+b,torch.abs(a-b)],-1)
        else:features=columns.new_empty((0,2*columns.shape[-1]))
        return [(self.keys[i](x)*self.query[i](context)).sum(-1)/math.sqrt(self.width)+self.bias[i]
                for i,x in enumerate([tables,columns,features])]

def load(group,training=False,reference=False):
    folder=OUT/group/'runtime';folder.mkdir(parents=True,exist_ok=True);pilot.OUT=folder
    if training:
        model,tok=pilot.load(train=True)
        if group!='structure':
            from safetensors.torch import load_file
            state=load_file(str(SOURCE/'D/adapter/adapter_model.safetensors'))
            set_peft_model_state_dict(model,state)
        return model,tok
    return pilot.load(train=False,adapter=SOURCE/'D/adapter' if reference else OUT/group/'adapter')

def model_logp(model,tok,row,context=False):
    batch={k:v.to('cuda') for k,v in collate([row],tok.pad_token_id).items()};labels=batch.pop('labels')
    positions=torch.nonzero(labels[0,1:]!=-100).flatten();targets=labels[0,positions+1];saved=[]
    handle=None
    if context:
        # Only the prompt-end state is given to the head. Causal attention prevents
        # it from seeing reference SQL, even though SQL CE shares the forward.
        index=len(row['prompt_ids'])-1
        handle=model.get_base_model().model.norm.register_forward_hook(lambda m,inp,out:saved.append(out[0,index].clone()))
    try:
        with sdpa_kernel(SDPBackend.MATH):
            logits=model(**batch,use_cache=False,logits_to_keep=positions).logits[0]
            result=completion_logp(logits,targets)
    finally:
        if handle is not None:handle.remove()
    return result,(saved[0] if saved else None)

def reference():
    assert read(OUT/'config.json')==CONFIG
    path=OUT/'reference-logps.jsonl';done={r['id']:r for r in records(path)};prefs=read(OUT/'preferences.json');model=tok=None
    for r in prefs:
        if r['id'] in done:continue
        if model is None:model,tok=load('reference',reference=True);model.eval()
        with torch.no_grad():
            a,_=model_logp(model,tok,r['chosen']);b,_=model_logp(model,tok,r['rejected'])
        assert bool(torch.isfinite(a)) and bool(torch.isfinite(b))
        record=dict(id=r['id'],chosen=float(a),rejected=float(b),reference='D adapter',chosen_tokens=sum(x!=-100 for x in r['chosen']['labels']),rejected_tokens=sum(x!=-100 for x in r['rejected']['labels']))
        append(path,record);done[r['id']]=record
        print('REFERENCE',len(done),len(prefs),flush=True)
    export_records(path);save(OUT/'reference-summary.json',dict(completed=True,count=len(done),reference='D adapter, never disabled adapter/base model'))

def schema_features(model):
    result={}
    with torch.no_grad():
        for db,r in read(OUT/'schema-candidates.json').items():
            embeddings=torch.stack([model.get_input_embeddings()(torch.tensor(ids,device='cuda')).float().mean(0) for ids in r['candidate_ids']]).detach()
            n=len(r['tables']);result[db]=(embeddings[:n],embeddings[n:])
    return result

def one_update(model,tok,heads,features,opt,sched,scaler,items,group):
    params=[p for p in model.parameters() if p.requires_grad]+(list(heads.parameters()) if heads is not None else [])
    opt.zero_grad(set_to_none=True);n=len(items);sql_total=tab_total=col_total=edge_total=pref_total=margin_total=0.;pair_correct=0
    denom=sum(sum(x!=-100 for x in (r if group=='structure' else r['chosen'])['labels'][1:]) for r in items)
    input_tokens=0;aux_counts=[[0,0,0] for _ in range(3)]
    for r in items:
        if group=='structure':
            logp,context=model_logp(model,tok,r,context=r['aux']['enabled']);ce=-logp/denom;loss=ce;sql_total+=float(ce.detach());input_tokens+=len(r['input_ids'])
            if r['aux']['enabled']:
                aux=r['aux'];tables,columns=features[r['db_id']];edges=torch.tensor(aux['edge_items'],device='cuda',dtype=torch.long).reshape(-1,2)
                logits=heads(context,tables,columns,edges);targets=[torch.tensor(aux[k],device='cuda') for k in ['table_labels','column_labels','edge_labels']]
                lt,lc,le=[balanced_bce(x,y) for x,y in zip(logits,targets)]
                loss=loss+(.1*(lt+lc)/2+.1*le)/n
                tab_total+=float(lt.detach())/n;col_total+=float(lc.detach())/n;edge_total+=float(le.detach())/n
                for j,(x,y) in enumerate(zip(logits,targets)):
                    pred=x.detach()>0;truth=y>0
                    aux_counts[j][0]+=int((pred&truth).sum());aux_counts[j][1]+=int(pred.sum());aux_counts[j][2]+=int(truth.sum())
        elif group=='sft-cont':
            logp,_=model_logp(model,tok,r['chosen']);loss=-logp/denom;sql_total+=float(loss.detach());input_tokens+=len(r['chosen']['input_ids'])
        else:
            cp,_=model_logp(model,tok,r['chosen']);rp,_=model_logp(model,tok,r['rejected'])
            preference,margin=dpo_loss(cp,rp,r['reference_chosen'],r['reference_rejected']);loss=preference/n
            pref_total+=float(preference.detach())/n;margin_total+=float(margin.detach())/n;pair_correct+=int(float(margin.detach())>0)
            input_tokens+=len(r['chosen']['input_ids'])+len(r['rejected']['input_ids'])
        assert bool(torch.isfinite(loss)),'Nonfinite loss'
        scaler.scale(loss).backward()
        # Do not keep forward graphs between accumulation microbatches.
        del loss
        if group=='structure':
            del logp,context,ce
            if r['aux']['enabled']:del logits,lt,lc,le
        elif group=='sft-cont':del logp
        else:del cp,rp,preference,margin
    scaler.unscale_(opt)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params),'Nonfinite gradient'
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters() if p.requires_grad),'No adapter gradient'
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad),'Frozen gradient'
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);lr=opt.param_groups[0]['lr'];before=scaler.get_scale()
    scaler.step(opt);scaler.update();assert scaler.get_scale()>=before,'Skipped update';sched.step();torch.cuda.synchronize()
    return dict(sql_loss=sql_total,table_loss=tab_total,column_loss=col_total,edge_loss=edge_total,dpo_loss=pref_total,preference_margin=margin_total,
        preference_accuracy=pair_correct/n if group=='dpo' else None,auxiliary_tp_pred_gold=aux_counts,
        loss=pref_total if group=='dpo' else sql_total+.1*(tab_total+col_total)/2+.1*edge_total,
        samples=n,input_tokens=input_tokens,chosen_supervised_tokens=denom,grad_norm=float(norm),learning_rate=lr,scale=scaler.get_scale(),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)

def save_checkpoint(folder,model,heads,opt,sched,scaler,group,step,history,ids):
    folder.mkdir(parents=True,exist_ok=True)
    state=dict(config=CONFIG,group=group,ids=ids,step=step,history=history,
        adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},heads=heads.state_dict() if heads else None,
        optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()))
    tmp=folder/'state.pt.tmp';torch.save(state,tmp);tmp.replace(folder/'state.pt');save(folder/'complete.json',dict(step=step,group=group,state='state.pt',hash_checks=False))

def train(group,profile=False):
    assert group in GROUPS and read(OUT/'config.json')==CONFIG
    folder=OUT/group/('profile' if profile else '');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():assert read(folder/'summary.json')['completed'];return
    if not profile:assert read(OUT/group/'profile/summary.json')['completed']
    if group=='structure':
        aux={r['id']:r for r in read(OUT/'structure-labels.json')};items=[dict(r,aux=aux[r['id']]) for r in read(OUT/'train.json')];cfg=CONFIG['structure']
    else:
        cfg=CONFIG['preference'];items=read(OUT/'preferences.json')
        if group=='dpo':
            rr={r['id']:r for r in records(OUT/'reference-logps.jsonl')};assert len(rr)==len(items)
            items=[dict(r,reference_chosen=rr[r['id']]['chosen'],reference_rejected=rr[r['id']]['rejected']) for r in items]
    accumulation=cfg['accumulation'];total=len(items)//accumulation;ids=[r['id'] for r in items]
    if profile:
        total=2;longest=max(items,key=lambda r:len(r['input_ids']) if group=='structure' else len(r['chosen']['input_ids'])+len(r['rejected']['input_ids']));items=[longest]*(total*accumulation)
    model,tok=load(group,training=True);model.train();heads=None;features=None
    if group=='structure':
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(3407);heads=StructureHeads(model.config.hidden_size).cuda().float()
        features=schema_features(model)
    import bitsandbytes as bnb
    from transformers import get_cosine_schedule_with_warmup
    params=[p for p in model.parameters() if p.requires_grad]+(list(heads.parameters()) if heads is not None else [])
    opt=bnb.optim.AdamW8bit(params,lr=cfg['lr'],weight_decay=.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.) if profile else get_cosine_schedule_with_warmup(opt,num_warmup_steps=cfg['warmup'],num_training_steps=total)
    scaler=torch.amp.GradScaler('cuda',init_scale=256);history=[];start=0
    if not profile:
        saved=sorted((folder/'checkpoints').glob('step-*/complete.json'))
        if saved:
            state=torch.load(saved[-1].parent/'state.pt',weights_only=False,map_location='cpu')
            assert state['config']==CONFIG and state['group']==group and state['ids']==ids
            set_peft_model_state_dict(model,state['adapter'])
            if heads is not None:heads.load_state_dict(state['heads'])
            opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
            start=state['step'];history=state['history'];del state
    begun=time.perf_counter()
    try:
        for step in range(start,total):
            before=time.perf_counter();m=one_update(model,tok,heads,features,opt,sched,scaler,items[step*accumulation:(step+1)*accumulation],group)
            m.update(step=step+1,seconds=time.perf_counter()-before);history.append(m);save(folder/'metrics.json',history)
            print('STEP',group,'profile' if profile else 'formal',step+1,total,'loss',m['loss'],'seconds',m['seconds'],flush=True)
            elapsed=time.perf_counter()-begun;eta=elapsed/(step+1-start)*(total-step-1)
            (folder/'progress.txt').write_text(f'{step+1}/{total} ({100*(step+1)/total:.1f}%) elapsed={elapsed:.1f}s remaining_estimate={eta:.1f}s','utf-8')
            if not profile and ((step+1)%10==0 or step+1==total):save_checkpoint(folder/'checkpoints'/f'step-{step+1:04d}',model,heads,opt,sched,scaler,group,step+1,history,ids)
        if not profile:
            model.save_pretrained(folder/'adapter',safe_serialization=True);tok.save_pretrained(folder/'adapter')
            if heads is not None:torch.save(heads.state_dict(),folder/'structure-heads.pt')
        save(folder/'summary.json',dict(completed=True,profile=profile,group=group,steps=total,samples=sum(r['samples'] for r in history),
            seconds=sum(r['seconds'] for r in history),input_tokens=sum(r['input_tokens'] for r in history),
            trainable_parameters=sum(p.numel() for p in params),head_parameters=sum(p.numel() for p in heads.parameters()) if heads else 0,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,skipped_updates=0,hash_checks=False,independent_review=False))
    except BaseException as e:save(folder/'failure.json',dict(error=repr(e),completed_steps=len(history)));raise

def infer(group):
    assert read(OUT/group/'summary.json')['completed'];model=tok=None;summaries={}
    for split,count in [('main',100),('financial',106)]:
        path=OUT/group/f'{split}-predictions.jsonl';prompts=read(OUT/f'{split}-prompts.json');done={r['id']:r for r in records(path)}
        for r in prompts:
            if r['context_covered'] or r['id'] in done:continue
            p=dict(id=r['id'],db_id=r['db_id'],raw_output='',pred_sql='',completed=False,seconds=0,generated_tokens=0,token_ids=[],generation_status='context_limit',split=split)
            append(path,p);done[r['id']]=p
        for pair in batches(prompts):
            if all(r['id'] in done for r in pair):continue
            if model is None:model,tok=load(group)
            for p in generate_batch(model,tok,pair):
                if p['id'] in done:continue
                p.update(group=group,split=split);append(path,p);done[p['id']]=p
            print('INFER',group,split,len(done),count,flush=True)
        assert len(done)==count;export_records(path)
        summaries[split]=dict(count=count,seconds=sum(r['seconds'] for r in done.values()),tokens=sum(r['generated_tokens'] for r in done.values()))
    save(OUT/group/'generation-summary.json',dict(completed=True,splits=summaries))

def pipeline():
    OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    jobs=[('prepare',None),('profile','structure'),('train','structure'),('reference',None),('profile','sft-cont'),('profile','dpo'),('train','sft-cont'),('train','dpo')]
    jobs += [('infer',g) for g in GROUPS]+[('score',g) for g in GROUPS]+[('report',None)]
    for stage,group in jobs:
        name=f'{int(time.time())}-{uuid.uuid4().hex[:6]}';log=OUT/'logs'/f'{name}-{stage}-{group}.log';before=time.time()
        save(OUT/'status.json',dict(status='running',stage=stage,group=group,log=str(log)))
        script='structure_dpo_score.py' if stage in ['score','report'] else 'structure_dpo.py'
        cmd=[sys.executable,'-u',str(ROOT/script),stage]+(['--group',group] if group else [])
        with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        append(OUT/'stage-costs.jsonl',dict(id=name,stage=stage,group=group,seconds=time.time()-before,returncode=r.returncode));export_records(OUT/'stage-costs.jsonl')
        if r.returncode:save(OUT/'status.json',dict(status='failed',stage=stage,group=group,log=str(log)));raise SystemExit(r.returncode)
    save(OUT/'status.json',dict(status='completed',report='REPORT.md'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','reference','profile','train','infer','pipeline']);p.add_argument('--group',choices=GROUPS);a=p.parse_args()
    if a.stage in ['profile','train']:train(a.group,profile=a.stage=='profile')
    elif a.stage=='infer':infer(a.group)
    else:globals()[a.stage]()
