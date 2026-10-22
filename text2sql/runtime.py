"""Portable launchers for new runs using the released experimental components.

The historical metrics come from results/, not from an unexecuted new run.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text('utf-8').splitlines() if line.strip()]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def write_rows(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value),encoding='utf-8');temporary.replace(path)


def metadata_formats(metadata):
    structural={};checking={}
    for db,meta in metadata.items():
        tables={};check_tables={}
        for table,item in meta['tables'].items():
            name=item.get('name',table);columns={};types={}
            for column,value in item['columns'].items():
                value=dict(name=column,type=value) if isinstance(value,str) else value
                columns[column.lower()]=dict(name=value.get('name',column),type=value.get('type','TEXT'))
                types[value.get('name',column)]=value.get('type','TEXT')
            tables[table.lower()]=dict(name=name,columns=columns)
            check_tables[name]=dict(columns=types,pk=item.get('pk',[]))
        structural[db]=dict(tables=tables)
        checking[db]=dict(tables=check_tables,foreign_keys=meta.get('foreign_keys',[]))
    return structural,checking


def set_model(path,output):
    from . import model
    model.MODEL=Path(path);model.OUT=Path(output);model.OUT.mkdir(parents=True,exist_ok=True)
    return model


def prepare(args):
    from .schema import candidates,labels_for
    from .batching import encode
    model=set_model(args.model,Path(args.output)/'runtime');tok=model.tokenizer()
    structural,checking=metadata_formats(json.loads(Path(args.metadata).read_text('utf-8')))
    schemas={db:candidates(meta,tok) for db,meta in structural.items()}
    prepared=[];excluded=[];source=rows(args.input)
    assert len({r['id'] for r in source})==len(source),'Duplicate sample IDs'
    for index,row in enumerate(source):
        assert [m['role'] for m in row['messages']]==['system','user']
        try:
            encoded=encode(row['messages']+[dict(role='assistant',content='```sql\n'+row['sql']+'\n```')],tok)
            if len(encoded['prompt_ids'])>1792:raise ValueError('Prompt exceeds 1792; no truncation')
        except ValueError as error:
            excluded.append(dict(id=row['id'],reason=str(error)));continue
        try:aux=dict(enabled=True,**labels_for(row['sql'],structural[row['db_id']],schemas[row['db_id']],3407+index))
        except Exception as error:aux=dict(enabled=False,reason=str(error))
        prepared.append(dict(id=row['id'],db_id=row['db_id'],sql=row['sql'],**encoded,aux=aux))
    out=Path(args.output)
    for name,value in [('train.json',prepared),('schema-candidates.json',schemas),('metadata.json',checking),('excluded.json',excluded)]:write(out/name,value)
    print(json.dumps(dict(prepared=len(prepared),excluded=len(excluded),structure_enabled=sum(r['aux']['enabled'] for r in prepared))))


def train(args):
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup
    from .losses import StructureHeads
    from .joint import joint_update
    data=Path(args.prepared);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    training=json.loads((data/'train.json').read_text('utf-8'))
    if not training:raise ValueError('No training rows')
    config=dict(revision=REVISION,seed=3407,rank=16,alpha=16,lr=2e-4,kl=.1,structure=.1,
        accumulation=16,epochs=1,precision='NF4/FP16/FP32',ids=[r['id'] for r in training],model=str(Path(args.model).resolve()))
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Resume configuration mismatch')
    write(out/'config.json',config)
    model,tok=set_model(args.model,out/'runtime').load(train=True)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(3407);heads=StructureHeads(model.config.hidden_size).cuda().float()
    schemas=json.loads((data/'schema-candidates.json').read_text('utf-8'));features={}
    with torch.no_grad():
        for db,schema in schemas.items():
            embeddings=torch.stack([model.get_input_embeddings()(torch.tensor(ids,device='cuda')).float().mean(0) for ids in schema['candidate_ids']]).detach()
            n=len(schema['tables']);features[db]=(embeddings[:n],embeddings[n:])
    params=[p for p in model.parameters() if p.requires_grad]+list(heads.parameters())
    optimizer=bnb.optim.AdamW8bit(params,lr=2e-4,weight_decay=.01)
    steps=math.ceil(len(training)/16)
    scheduler=get_cosine_schedule_with_warmup(optimizer,min(2,max(0,steps-1)),steps)
    scaler=torch.amp.GradScaler('cuda',init_scale=256)
    start=0;history=[]
    latest=out/'resume.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Checkpoint configuration mismatch')
        set_peft_model_state_dict(model,state['adapter']);heads.load_state_dict(state['heads'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        random.setstate(state['python']);np.random.set_state(state['numpy']);start=state['step'];history=state['history']
    model.train()
    for step in range(start,steps):
        metric=joint_update(model,tok,heads,features,optimizer,scheduler,scaler,training[step*16:(step+1)*16])
        history.append(dict(step=step+1,**metric));write(out/'metrics.json',history);print(json.dumps(history[-1]),flush=True)
        if (step+1)%10==0 or step+1==steps:
            state=dict(config=config,step=step+1,history=history,adapter=get_peft_model_state_dict(model),heads=heads.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())
            temp=out/'resume.tmp';torch.save(state,temp);temp.replace(latest)
    model.save_pretrained(out/'adapter');tok.save_pretrained(out/'adapter');torch.save(heads.state_dict(),out/'structure-heads.pt')
    write(out/'summary.json',dict(completed=True,steps=steps,samples=len(training),peak_GiB=torch.cuda.max_memory_allocated()/1024**3))


def accepted(candidate,issues):
    return candidate['completed'] and candidate['generation_status']=='ok' and bool(candidate['pred_sql'].strip()) and not any(i['severity']=='error' for i in issues)


def infer(args):
    from .decoding import generate
    from .checks import inspect_sql
    from .feedback import diagnostics,fresh_messages
    metadata=json.loads(Path(args.metadata).read_text('utf-8'));_,metadata=metadata_formats(metadata)
    out=Path(args.output);binding=dict(adapter=str(args.adapter),model=str(args.model),repair=args.repair,input=str(Path(args.input).resolve()),metadata=str(Path(args.metadata).resolve()))
    manifest=out.with_suffix('.binding.json')
    if manifest.exists() and json.loads(manifest.read_text())!=binding:raise ValueError('Prediction cache belongs to another run')
    write(manifest,binding)
    model,tok=set_model(args.model,out.parent/'runtime').load(adapter=args.adapter)
    original=rows(args.input);done={r['id']:r for r in rows(out)} if out.exists() else {}
    if not set(done)<={r['id'] for r in original}:raise ValueError('Unknown cached ID')
    prompts=[]
    for r in original:
        assert [m['role'] for m in r['messages']]==['system','user']
        text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(text,add_special_tokens=False)['input_ids'];prompts.append(dict(r,prompt_ids=ids,context_covered=len(ids)<=1792))
    for r in prompts:
        if not r['context_covered'] and r['id'] not in done:
            done[r['id']]=dict(id=r['id'],db_id=r['db_id'],pred_sql='',completed=False,generation_status='context_limit')
    pending=[r for r in prompts if r['context_covered'] and r['id'] not in done]
    for start in range(0,len(pending),2):
        pair=pending[start:start+2];generated=generate(model,tok,pair,metadata,'syntax-binding')
        byid={r['id']:r for r in pair}
        for result in generated:
            result['correction_attempted']=False;result['correction_accepted']=False
        if args.repair:
            repairs=[]
            for result in generated:
                issues=inspect_sql(result['pred_sql'],metadata[result['db_id']])
                if not result['completed']:issues.append(dict(code='incomplete',severity='error',message='Incomplete output'))
                if not issues:continue
                original_row=byid[result['id']]
                messages=fresh_messages(original_row['messages'],diagnostics(result['pred_sql'],metadata[result['db_id']],issues))
                text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok(text,add_special_tokens=False)['input_ids']
                if len(ids)<=1792:repairs.append(dict(original_row,messages=messages,prompt_ids=ids))
            if repairs:
                candidates=generate(model,tok,repairs,metadata,'syntax-binding');candidates={r['id']:r for r in candidates}
                for i,result in enumerate(generated):
                    candidate=candidates.get(result['id'])
                    if candidate is None:continue
                    issues=inspect_sql(candidate['pred_sql'],metadata[candidate['db_id']]);ok=accepted(candidate,issues)
                    final=dict(candidate if ok else result)
                    final.update(correction_attempted=True,correction_accepted=bool(ok),original=result,candidate=candidate,residual_issues=issues)
                    generated[i]=final
        for result in generated:done[result['id']]=result
        write_rows(out,[done[r['id']] for r in original if r['id'] in done])
    write_rows(out,[done[r['id']] for r in original])


def score(args):
    from .sqlite import execute_readonly,compare_rows
    from .metrics import result_sets_match
    prediction={r['id']:r for r in rows(args.predictions)};references=rows(args.references)
    assert set(prediction)=={r['id'] for r in references},'Keep all failures in the denominator'
    scored=[]
    for reference in references:
        pred=prediction[reference['id']]
        gold=execute_readonly(reference['db_path'],reference['sql'])
        if gold['status']!='ok':raise RuntimeError('Invalid reference SQL: '+reference['id'])
        result=execute_readonly(reference['db_path'],pred['pred_sql']) if pred['completed'] and pred['generation_status']=='ok' else {'status':pred['generation_status']}
        valid=result['status']=='ok'
        comparison=compare_rows(result['rows'],gold['rows']) if valid else dict(ex=False,bag=False)
        scored.append(dict(id=reference['id'],db_id=reference['db_id'],valid=valid,status=result['status'],
            repository_ex=valid and result_sets_match(result['rows'],gold['rows']),official_set_ex=comparison['ex'],bag_ex=comparison['bag']))
    write_rows(args.output,scored)
    print(json.dumps(dict(n=len(scored),correct=sum(r['repository_ex'] for r in scored),executable=sum(r['valid'] for r in scored))))


def demo():
    from .schema import ast_labels
    from .checks import inspect_sql
    from .feedback import diagnostics
    root=Path(__file__).resolve().parents[1]
    structural,checking=metadata_formats(json.loads((root/'examples/schema.json').read_text()))
    sql='SELECT c.name FROM client c JOIN account a ON c.client_id = a.account_id'
    issues=inspect_sql(sql,checking['toy'])
    print(json.dumps(dict(sql=sql,ast=ast_labels(sql,structural['toy']),issues=issues,
        feedback=diagnostics(sql,checking['toy'],issues)),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo')
    p=sub.add_parser('prepare');p.add_argument('--input',required=True);p.add_argument('--metadata',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train');p.add_argument('--prepared',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('infer');p.add_argument('--input',required=True);p.add_argument('--metadata',required=True);p.add_argument('--model',required=True);p.add_argument('--adapter');p.add_argument('--output',required=True);p.add_argument('--repair',action='store_true')
    p=sub.add_parser('score');p.add_argument('--predictions',required=True);p.add_argument('--references',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='demo':demo()
    else:globals()[args.command](args)
