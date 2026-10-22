"""D-data SFT + full-vocabulary forward KL; no hash checks or review stages."""
import argparse,json,math,os,random,shutil,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import get_peft_model_state_dict,set_peft_model_state_dict
import pilot
from pilot import read,save,collate
from experiment import records,append,export_records
from hardpair_batch2 import batches,generate_batch
from generalization_runtime import summarize,contrast

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/D-kl-v1'
TRAIN=ROOT/'experiments/data-mixture-v1'
EVAL=ROOT/'experiments/hardpair-v1/inference-batch2-v1'
SPLITS={'main':100,'financial':106}
CONFIG=dict(model='Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    initialization='fresh base + fresh LoRA, D data/recipe reused, not continued from D adapter',
    seed=3407,samples=800,steps=50,epochs=1,learning_rate=2e-4,warmup_steps=2,
    micro_batch=1,accumulation=16,rank=16,alpha=16,dropout=0,weight_decay=.01,
    optimizer='AdamW8bit',scheduler='cosine',clip=1.,initial_scale=256,
    precision='NF4 double / FP16 matrix compute / FP32 adapter and unquantized',backend='SDPA MATH',
    kl_weight=1.,temperature=1.,kl_direction='base || adapted',
    kl_context='same D reference-SQL teacher-forced prefixes, assistant and EOS positions only',
    kl_vocabulary='full vocabulary, FP32 log softmax, no top-k approximation',
    kl_normalization='sum over vocabulary and supervised positions, divided by accumulation-window supervised token count',
    teacher='same quantized backbone with adapter disabled, eval mode, no_grad; no reference eval answers',
    train_limit=2048,prompt_limit=1792,eval_batch=2,max_new_tokens=512,
    do_sample=False,thinking=False,evaluation='BIRD frozen main100 + financial106, original author prompt, batch2, no correction',hash_checks=False,independent_review=False)

DEFAULT_CONFIG=dict(CONFIG)
DEFAULT_OUT=OUT

def configure(weight=1.,output_dir=None):
    global CONFIG,OUT
    if not math.isfinite(weight) or weight<0:raise ValueError('KL weight must be finite and nonnegative')
    if weight!=1. and output_dir is None:raise ValueError('Nondefault KL weight requires an explicit output directory')
    target=Path(output_dir).resolve() if output_dir else DEFAULT_OUT
    config=dict(DEFAULT_CONFIG,kl_weight=float(weight))
    if (target/'config.json').exists() and read(target/'config.json')!=config:
        raise ValueError('Output directory belongs to a different configuration')
    CONFIG=config;OUT=target

def window_loss(ce,kl,denominator,weight):
    assert denominator>0
    return (ce+weight*kl)/denominator

def validate_resume(state,ids):
    if state['config']!=CONFIG or state['train_ids']!=ids:
        raise ValueError('Checkpoint configuration or training order mismatch')

def loss_sums(student_logits,teacher_logp,targets,chunk=16):
    """Both outputs are sums across the same shifted assistant positions."""
    ce=student_logits.new_zeros((),dtype=torch.float32);kl=ce.clone()
    for start in range(0,len(targets),chunk):
        logits=student_logits[start:start+chunk].float()
        logq=F.log_softmax(logits,dim=-1)
        logp=teacher_logp[start:start+chunk].detach()
        ce=ce+F.nll_loss(logq,targets[start:start+chunk],reduction='sum')
        kl=kl+F.kl_div(logq,logp,log_target=True,reduction='sum')
    return ce,kl

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'config.json').exists():
        assert read(OUT/'config.json')==CONFIG;return
    rows=read(TRAIN/'frozen/D/train.json');prompts=read(EVAL/'main-prompts.json')
    assert len(rows)==800 and len(prompts)==100
    assert not {r['db_id'] for r in rows}&{r['db_id'] for r in prompts}
    assert len({r['id'] for r in rows})==800
    for r in rows:
        n=len(r['prompt_ids']);assert n<=1792 and len(r['input_ids'])<=2048
        assert r['labels'][:n]==[-100]*n and r['labels'][n:]==r['input_ids'][n:]
    shutil.copy2(TRAIN/'frozen/D/train.json',OUT/'train.json')
    for split,count in SPLITS.items():
        pp=read(EVAL/f'{split}-prompts.json');assert len(pp)==count
        assert not {r['db_id'] for r in rows}&{r['db_id'] for r in pp}
        shutil.copy2(EVAL/f'{split}-prompts.json',OUT/f'{split}-prompts.json')
    save(OUT/'config.json',CONFIG)
    save(OUT/'data-summary.json',dict(count=len(rows),databases=len({r['db_id'] for r in rows}),
        input_tokens=sum(len(r['input_ids']) for r in rows),supervised_tokens=sum(sum(v!=-100 for v in r['labels'][1:]) for r in rows),
        source=str(TRAIN/'frozen/D/train.json'),order='exact source order',evaluation=CONFIG['evaluation'],
        note=f'No additional preservation dataset or teacher generation; KL uses existing SQL-prefix contexts. Lambda fixed at {CONFIG["kl_weight"]} before this run, no hyperparameter sweep.'))
    print('PREPARED',flush=True)

def load(train=False):
    pilot.OUT=OUT/('train-runtime' if train else 'infer-runtime');pilot.OUT.mkdir(exist_ok=True)
    return pilot.load(train=train,adapter=None if train else OUT/'adapter')

def update(model,tok,opt,sched,scaler,rows):
    params=[p for p in model.parameters() if p.requires_grad];opt.zero_grad(set_to_none=True)
    denominator=sum(sum(v!=-100 for v in r['labels'][1:]) for r in rows);assert denominator>0
    ce_total=kl_total=teacher_seconds=student_seconds=0.
    for row in rows:
        batch={k:v.to('cuda') for k,v in collate([row],tok.pad_token_id).items()};labels=batch.pop('labels')
        positions=torch.nonzero(labels[0,1:]!=-100).flatten();targets=labels[0,positions+1]
        start=time.perf_counter();model.eval()
        with torch.no_grad(),model.disable_adapter(),sdpa_kernel(SDPBackend.MATH):
            teacher_logits=model(**batch,use_cache=False,logits_to_keep=positions).logits[0]
            logp=F.log_softmax(teacher_logits.float(),dim=-1).detach()
        del teacher_logits;torch.cuda.synchronize();teacher_seconds+=time.perf_counter()-start
        model.train();start=time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            logits=model(**batch,use_cache=False,logits_to_keep=positions).logits[0]
            ce,kl=loss_sums(logits,logp,targets)
            loss=window_loss(ce,kl,denominator,CONFIG['kl_weight'])
        assert bool(torch.isfinite(ce)) and bool(torch.isfinite(kl)) and bool(torch.isfinite(loss)),row['id']
        assert float(kl.detach())>=-1e-3,'Unexpected negative KL'
        ce_total+=float(ce.detach())/denominator;kl_total+=float(kl.detach())/denominator
        scaler.scale(loss).backward()
        del logits,logp,loss,ce,kl,batch,labels,targets,positions
        torch.cuda.synchronize();student_seconds+=time.perf_counter()-start
    scaler.unscale_(opt)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params),'Nonfinite gradient'
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in params),'Zero gradients'
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    lr=opt.param_groups[0]['lr'];scale=scaler.get_scale();scaler.step(opt);scaler.update()
    assert scaler.get_scale()>=scale,'Skipped optimizer update';sched.step()
    return dict(loss=ce_total+CONFIG['kl_weight']*kl_total,sql_loss=ce_total,kl_loss=kl_total,
        weighted_kl=CONFIG['kl_weight']*kl_total,grad_norm=float(norm),scale=scaler.get_scale(),learning_rate=lr,
        next_learning_rate=opt.param_groups[0]['lr'],samples=len(rows),supervised_tokens=denominator,
        input_tokens=sum(len(r['input_ids']) for r in rows),teacher_forward_seconds=teacher_seconds,student_forward_backward_seconds=student_seconds,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)

def checkpoint(folder,model,opt,sched,scaler,step,history,ids):
    folder.mkdir(parents=True,exist_ok=True)
    state=dict(adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()},optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()),
        config=CONFIG,step=step,position=16*step,train_ids=ids,history=history)
    temp=folder/'state.pt.tmp';torch.save(state,temp);temp.replace(folder/'state.pt')
    save(folder/'complete.json',dict(step=step,position=step*16,state='state.pt',hash_checks=False))

def train(profile=False):
    assert read(OUT/'config.json')==CONFIG
    folder=OUT/'profile' if profile else OUT;folder.mkdir(exist_ok=True)
    if (folder/'train-summary.json').exists():assert read(folder/'train-summary.json')['completed'];return
    if not profile:assert read(OUT/'profile/train-summary.json')['completed']
    rows=read(OUT/'train.json');ids=[r['id'] for r in rows];total=3 if profile else 50
    if profile:rows=[max(rows,key=lambda r:len(r['input_ids']))]*48
    model,tok=load(train=True);model.train()
    import bitsandbytes as bnb
    from transformers import get_cosine_schedule_with_warmup
    opt=bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.) if profile else get_cosine_schedule_with_warmup(opt,num_warmup_steps=2,num_training_steps=50)
    scaler=torch.amp.GradScaler('cuda',init_scale=256);history=[];start_step=0
    if not profile:
        checkpoints=sorted((OUT/'checkpoints').glob('step-*/complete.json'))
        if checkpoints:
            state=torch.load(checkpoints[-1].parent/'state.pt',map_location='cpu',weights_only=False)
            validate_resume(state,ids)
            set_peft_model_state_dict(model,state['adapter']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            torch.set_rng_state(state['rng']['cpu']);torch.cuda.set_rng_state_all(state['rng']['cuda']);random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
            start_step=state['step'];history=state['history'];del state
    begun=time.perf_counter()
    try:
        for step in range(start_step,total):
            start=time.perf_counter();m=update(model,tok,opt,sched,scaler,rows[step*16:(step+1)*16]);m.update(step=step+1,position=(step+1)*16,seconds=time.perf_counter()-start)
            history.append(m);save(folder/'metrics.json',history)
            print('KL_STEP',('profile' if profile else 'formal'),step+1,total,'SQL',m['sql_loss'],'KL',m['kl_loss'],'seconds',m['seconds'],flush=True)
            if not profile and step==0:
                save(OUT/'startup-ready.json',dict(passed=True,completed_steps=1,kl_weight=CONFIG['kl_weight'],
                    loss=m['loss'],sql_loss=m['sql_loss'],kl_loss=m['kl_loss'],weighted_kl=m['weighted_kl'],
                    finite_update=True,note='Normal startup confirmed; no continued external polling required.'))
            elapsed=time.perf_counter()-begun;eta=elapsed/max(1,step+1-start_step)*(total-step-1)
            (folder/'progress.txt').write_text(f'{step+1}/{total} ({100*(step+1)/total:.1f}%) elapsed={elapsed:.1f}s estimated_remaining={eta:.1f}s','utf-8')
            if not profile and (step+1)%10==0:checkpoint(OUT/'checkpoints'/f'step-{step+1:04d}',model,opt,sched,scaler,step+1,history,ids)
        if not profile:model.save_pretrained(OUT/'adapter',safe_serialization=True);tok.save_pretrained(OUT/'adapter')
        save(folder/'train-summary.json',dict(completed=True,profile=profile,steps=total,samples=total*16,config=CONFIG,skipped_updates=0,
            optimization_seconds=sum(m['seconds'] for m in history),teacher_forward_seconds=sum(m['teacher_forward_seconds'] for m in history),
            student_forward_backward_seconds=sum(m['student_forward_backward_seconds'] for m in history),
            input_tokens=sum(m['input_tokens'] for m in history),supervised_tokens=sum(m['supervised_tokens'] for m in history),
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
            note='No parameter hash/reload replay/independent audit at user request. All frozen parameters require_grad=False; nonfinite loss/gradients or skipped update abort.'))
    except BaseException as error:
        save(folder/'failure.json',dict(error=repr(error),completed_steps=len(history)));raise

def infer():
    assert read(OUT/'train-summary.json')['completed'];model=tok=None;summaries={}
    for split,count in SPLITS.items():
        path=OUT/f'{split}-predictions.jsonl';rows=read(OUT/f'{split}-prompts.json');done={r['id']:r for r in records(path)}
        for r in rows:
            if r['context_covered'] or r['id'] in done:continue
            p=dict(id=r['id'],db_id=r['db_id'],raw_output='',pred_sql='',completed=False,seconds=0,generated_tokens=0,token_ids=[],generation_status='context_limit',split=split)
            append(path,p);done[r['id']]=p
        for pair in batches(rows):
            if all(r['id'] in done for r in pair):continue
            if model is None:model,tok=load()
            for p in generate_batch(model,tok,pair):
                if p['id'] in done:continue
                p.update(experiment=OUT.name,kl_weight=CONFIG['kl_weight'],split=split);append(path,p);done[p['id']]=p
            print('INFER',split,len(done),count,flush=True)
        assert len(done)==count;export_records(path)
        summaries[split]=dict(count=count,seconds=sum(r['seconds'] for r in done.values()),tokens=sum(r['generated_tokens'] for r in done.values()))
    save(OUT/'generation-summary.json',dict(completed=True,splits=summaries,seconds=sum(r['seconds'] for r in summaries.values())))

def score():
    assert read(OUT/'generation-summary.json')['completed']
    from reproduction_sqlite import execute_readonly,compare_rows
    from src.evaluate import result_sets_match
    for split,count in SPLITS.items():
        refs={r['id']:r for r in read(EVAL/f'evaluation-only/{split}-references.json')};path=OUT/f'{split}-scores.jsonl';done={r['id']:r for r in records(path)}
        for p in records(OUT/f'{split}-predictions.jsonl'):
            i=p['id']
            if i in done:continue
            r=refs[i];gold=execute_readonly(r['db_path'],r['sql']);assert gold['status']=='ok'
            covered=p['generation_status']!='context_limit'
            pred=execute_readonly(r['db_path'],p['pred_sql']) if p['completed'] and p['pred_sql'].strip() else dict(status='output_incomplete' if covered else 'context_limit')
            valid=pred['status']=='ok';cmp=compare_rows([tuple(x) for x in pred.get('rows',[])],[tuple(x) for x in gold['rows']]) if valid else dict(ex=False,bag=False)
            record=dict(id=i,db_id=r['db_id'],valid=valid,status=pred['status'],error=pred.get('error',''),repository_ex=valid and result_sets_match(pred['rows'],gold['rows']),official_set_ex=cmp['ex'],bag_ex=cmp['bag'],context_covered=covered)
            append(path,record);done[i]=record;print('SCORE',split,i,record['repository_ex'],flush=True)
        assert len(done)==count;export_records(path)

def report():
    results={};diffs={};scores={}
    for split,count in SPLITS.items():
        allscores=dict(baseline=records(EVAL/f'baseline/{split}-scores.jsonl'),D=records(EVAL/f'D/{split}-scores.jsonl'),KL=records(OUT/f'{split}-scores.jsonl'))
        assert all(len(v)==count for v in allscores.values());scores[split]=allscores
        results[split]={g:summarize(v) for g,v in allscores.items()}
        diffs[split]={f'KL-{g}':contrast(v,allscores['KL']) for g,v in allscores.items() if g!='KL'}
    fs={g:{r['id']:r for r in rr} for g,rr in scores['financial'].items()}
    oldreg=[i for i,r in fs['baseline'].items() if r['repository_ex'] and not fs['D'][i]['repository_ex']]
    recovery=dict(original_D_regressions=oldreg,recovered=[i for i in oldreg if fs['KL'][i]['repository_ex']],remaining=[i for i in oldreg if not fs['KL'][i]['repository_ex']])
    s=read(OUT/'train-summary.json');old=read(TRAIN/'D/train-summary.json');gen=read(OUT/'generation-summary.json')
    save(OUT/'comparison.json',dict(groups=results,contrasts=diffs,financial_recovery=recovery,training=s,generation=gen,D_training_seconds=old['optimization_seconds'],stages=records(OUT/'stage-costs.jsonl')))
    lines=['# D 数据与配方：SFT＋基座 KL 对照','',
        '使用D原800条BIRD训练数据（含D原定向增强）、原顺序，从基础模型重新初始化LoRA；没有继续训练旧D adapter。普通SFT对照复用D既有50步训练与BIRD batch2预测。唯一计划训练目标变化为SQL CE + 1.0×KL(base || adapted)，温度1，全词表。KL基于相同参考SQL前缀，仅assistant/EOS位置；按每个累积窗口监督token总数归一化。没有额外保留集，也未根据评测成绩选择KL系数。','']
    for split in SPLITS:
        lines += [f'## {split}','','|组别|EX|可执行率|set EX|重复行敏感|','|---|---:|---:|---:|---:|']
        for g,r in results[split].items():lines.append(f"|{g}|{r['ex']:.1%}|{r['valid']:.1%}|{r['set_ex']:.1%}|{r['bag_ex']:.1%}|")
        for k,r in diffs[split].items():lines.append(f"\n{k}: {100*r['delta']:+.1f}pp；修复{len(r['repairs'])}题、退步{len(r['harms'])}题。"+(f"数据库聚类bootstrap95%区间[{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}]pp。" if split=='main' else '仅一个数据库，不使用退化的数据库聚类区间作推断。'))
    lines += ['',f"financial原D退步{len(oldreg)}题，本轮恢复{len(recovery['recovered'])}题，仍错{len(recovery['remaining'])}题；仅指执行分数恢复，不等于语义正确性证明。"]
    lines += ['',f"KL正式训练{s['optimization_seconds']/60:.2f}分钟，其中基座无梯度前向{s['teacher_forward_seconds']/60:.2f}分钟；历史D训练{old['optimization_seconds']/60:.2f}分钟。同样50优化步，但KL增加计算量，并非相同FLOPs。峰值已分配显存{s['peak_allocated_GiB']:.2f}GiB；零跳过更新。生成{gen['seconds']:.2f}秒。",'',
        '模型Qwen3-4B-Instruct-2507；NF4/FP16、FP32 adapter、rank16/alpha16、AdamW8bit、weight decay .01、clip1、GradScaler256、seed3407、microbatch1/累积16、cosine/2步warmup、峰值2e-4、1epoch/50步均沿用D。基座与学生共享量化骨干，教师关闭adapter/eval/no_grad，教师分布停止梯度；学生启用adapter训练。', '',
        '本轮只做原始模型生成，不加静态纠错。不使用Spider；复用原作者使用的BIRD数据集上的既有主评测100题/8库与financial106辅助题。两份均已查看，是开发实验，不是完整BIRD官方成绩。原提示词、batch2、greedy512；主评测13道超长题继续计失败，覆盖87%。沿用作者仓库EX及同一只读执行器30秒/100000行/64MiB限制。单种子、单KL系数，不能推广为通用最优方法。', '',
        '按用户要求未执行哈希核对、额外重载一致性复验或代理复核。仅进行了新增损失的必要数值测试、最长样本3步显存/稳定性预检，以及正式训练的有限值/梯度保护。检查点包含adapter、优化器、scheduler、GradScaler、RNG、配置、数据ID顺序和位置；恢复只比较配置及ID，不核对哈希。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8');save(OUT/'status.json',dict(status='completed',report='REPORT.md',hash_checks=False,review=False));print('RESULT',results,flush=True)

def pipeline():
    OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for stage in ['prepare','profile','train','infer','score','report']:
        key=f'{int(time.time())}-{uuid.uuid4().hex[:6]}';log=OUT/'logs'/f'{key}-{stage}.log';start=time.time()
        save(OUT/'status.json',dict(status='running',stage=stage,log=str(log)))
        script=ROOT/'d_kl_score.py' if CONFIG['kl_weight']!=1. and stage in ['score','report'] else Path(__file__).resolve()
        args=[sys.executable,'-u',str(script),stage,'--kl-weight',str(CONFIG['kl_weight']),'--output-dir',str(OUT)]
        with log.open('w',encoding='utf-8') as f:r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        append(OUT/'stage-costs.jsonl',dict(id=key,stage=stage,seconds=time.time()-start,returncode=r.returncode));export_records(OUT/'stage-costs.jsonl')
        if r.returncode:save(OUT/'status.json',dict(status='failed',stage=stage,log=str(log)));raise SystemExit(r.returncode)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','profile','train','infer','score','report','pipeline'])
    parser.add_argument('--kl-weight',type=float,default=1.);parser.add_argument('--output-dir',type=Path);arg=parser.parse_args()
    configure(arg.kl_weight,arg.output_dir)
    if arg.stage in ['score','report'] and CONFIG['kl_weight']!=1.:
        import d_kl_score as scoring
        scoring.configure(OUT,CONFIG['kl_weight']);getattr(scoring,arg.stage)()
    else:train(profile=True) if arg.stage=='profile' else globals()[arg.stage]()
