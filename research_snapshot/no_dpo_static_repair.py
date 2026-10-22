"""One schema-only regeneration after the frozen no-DPO constrained output."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pilot
from pilot import read,save
from structure_dpo_score import records,put,export
from static_repair_check import inspect_sql
from regenerate_feedback import diagnostics,fresh_messages

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'experiments/combined-ablation-v1'
SOURCE=OLD/'groups/no-dpo'
ADAPTER=OLD/'runs/joint-sft/adapter'
EVAL=ROOT/'experiments/hardpair-v1/inference-batch2-v1'
OUT=ROOT/'experiments/no-dpo-static-repair-v1'
SPLITS={'main':100,'financial':106}
CONFIG=dict(experiment='no-dpo-static-repair-v1',source=str(SOURCE),adapter=str(ADAPTER),
    revision='cdbee75f17c01a7cc42f958dc650907174af0554',seed=3407,batch_size=2,
    max_new_tokens=512,prompt_limit=1792,do_sample=False,thinking=False,decode='syntax-binding',
    attempts=1,trigger='any existing static issue including warnings; context failures excluded',
    prompt='fresh system/user; original schema/question/evidence and diagnostics; no old SQL',
    accept='complete, status ok, nonempty SQL and no static error; warnings allowed',
    fallback='retain original output',hash_checks=False,extra_review=False)


def validate(actual,expected):
    if actual!=expected:raise ValueError('Configuration, source, group or ID order mismatch')


def freeze(path,value):
    if path.exists():validate(read(path),value)
    else:save(path,value)


def binding(group='corrected'):
    assert group=='corrected'
    return dict(config=CONFIG,group=group,adapter=str(ADAPTER),decode='syntax-binding',
        ids={s:[r['id'] for r in read(OUT/f'{s}-prompts.json')] for s in SPLITS})


def issues_for(pred,meta):
    if pred['generation_status']=='context_limit':return []
    issues=inspect_sql(pred['pred_sql'],meta)
    if not pred['completed']:
        issues.append(dict(code='incomplete',severity='error',message='Previous SQL output did not finish. Return one complete query without repetition.'))
    return issues


def make_prompt(row,feedback,tok):
    messages=fresh_messages(row['messages'],feedback)
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok(text,add_special_tokens=False)['input_ids']
    return dict(id=row['id'],db_id=row['db_id'],messages=messages,prompt_ids=ids,context_covered=len(ids)<=1792)


def finalize(old,check,candidate=None,residual=None,postseconds=0):
    residual=[] if residual is None else residual
    accepted=(candidate is not None and candidate['completed'] and candidate['generation_status']=='ok'
        and bool(candidate['pred_sql'].strip()) and not any(i['severity']=='error' for i in residual))
    result=dict(candidate if accepted else old)
    result.update(group='corrected',adapter=str(ADAPTER),decode='syntax-binding',
        seconds=old['seconds'],original_generated_tokens=old.get('generated_tokens',0),
        correction_attempted=candidate is not None,correction_accepted=bool(accepted),
        correction_reason=('accepted' if accepted else 'candidate_rejected') if candidate is not None else check['reason'],
        sql_changed=bool(accepted and candidate['pred_sql']!=old['pred_sql']),
        repeated_sql=bool(candidate is not None and candidate['pred_sql'].strip()==old['pred_sql'].strip()),
        extra_generation_seconds=candidate['seconds'] if candidate is not None else 0.,
        extra_generated_tokens=candidate.get('generated_tokens',0) if candidate is not None else 0,
        extra_parser_seconds=candidate.get('parser_seconds',0) if candidate is not None else 0.,
        residual_issues=residual,postcheck_seconds=postseconds)
    return result


def pending_batches(rows,done):
    # Fixed eligibility pairing, with cached candidates never regenerated.
    for i in range(0,len(rows),2):
        pending=[r for r in rows[i:i+2] if r['id'] not in done]
        if pending:yield pending


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);freeze(OUT/'config.json',CONFIG)
    if (OUT/'prepared.json').exists():return
    tok=pilot.tokenizer()
    meta=read(ROOT/'experiments/static-regenerate-v1/metadata.json')
    freeze(OUT/'check-metadata.json',meta)
    freeze(OUT/'metadata.json',read(OLD/'metadata.json'))
    shutil.copyfile(OLD/'grammar.lark',OUT/'grammar.lark')
    counts={}
    for split,count in SPLITS.items():
        rows=read(OLD/f'{split}-prompts.json');original=records(SOURCE/f'{split}-predictions.jsonl')
        assert len(rows)==len(original)==count
        byid={r['id']:r for r in original};assert set(byid)=={r['id'] for r in rows}
        assert sum(not r['context_covered'] for r in rows)==(13 if split=='main' else 0)
        freeze(OUT/f'{split}-prompts.json',rows)
        checks=[];prompts=[]
        for row in rows:
            pred=byid[row['id']];begun=time.perf_counter();issues=issues_for(pred,meta[row['db_id']])
            feedback=diagnostics(pred['pred_sql'],meta[row['db_id']],issues) if issues else ''
            prompt=make_prompt(row,feedback,tok) if issues else None
            reason=('original_context_limit' if not row['context_covered'] else 'no_issue' if not issues else
                    'eligible' if prompt['context_covered'] else 'repair_context_limit')
            checks.append(dict(id=row['id'],db_id=row['db_id'],issues=issues,feedback=feedback,flagged=bool(issues),
                eligible=reason=='eligible',reason=reason,prompt_tokens=len(prompt['prompt_ids']) if prompt else 0,
                seconds=time.perf_counter()-begun))
            if reason=='eligible':prompts.append(prompt)
            freeze((OUT/'original'/f'{split}-predictions.jsonl').with_suffix('.records')/(pred['id']+'.json'),pred)
        for item in records(SOURCE/f'{split}-scores.jsonl'):
            freeze((OUT/'original'/f'{split}-scores.jsonl').with_suffix('.records')/(item['id']+'.json'),item)
        for kind in ['predictions','scores']:export(OUT/'original'/f'{split}-{kind}.jsonl')
        freeze(OUT/f'{split}-checks.json',checks)
        freeze(OUT/f'{split}-repair-prompts.json',prompts)
        counts[split]=dict(total=count,flagged=sum(c['flagged'] for c in checks),eligible=len(prompts),
                           reasons=dict(Counter(c['reason'] for c in checks)))
    freeze(OUT/'groups/corrected/binding.json',binding())
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['no_dpo_static_repair.py','test_no_dpo_static_repair.py','static_repair_check.py','regenerate_feedback.py']:
        shutil.copyfile(ROOT/name,code/name)
    save(OUT/'prepared.json',dict(completed=True,counts=counts,source=str(SOURCE),adapter=str(ADAPTER)))
    (OUT/'RUN_GUIDE.md').write_text('# no-DPO plus one static repair\n\n'
        'Run/resume: python no_dpo_static_repair.py pipeline\n'
        'Frozen syntax-binding rules, same no-DPO adapter, batch2 greedy512; no training.\n'
        'Original outputs and scores reused. One fresh prompt without old SQL per eligible ID.\n'
        'Stop external monitoring after startup-ready.json. Results: REPORT.md/comparison.json.\n',encoding='utf-8')
    print('PREPARED',counts,flush=True)


def infer():
    validate(read(OUT/'config.json'),CONFIG);b=binding();validate(read(OUT/'groups/corrected/binding.json'),b)
    import incremental_sql as inc
    inc.OUT=OUT
    meta=read(OUT/'metadata.json');checkmeta=read(OUT/'check-metadata.json')
    model=tok=None
    for split,count in SPLITS.items():
        root=OUT/'groups/corrected';path=root/f'{split}-predictions.jsonl';attemptpath=root/f'{split}-attempts.jsonl'
        old={r['id']:r for r in records(OUT/'original'/f'{split}-predictions.jsonl')}
        done={r['id']:r for r in records(path)};attempts={r['id']:r for r in records(attemptpath)}
        checks={r['id']:r for r in read(OUT/f'{split}-checks.json')};prompts=read(OUT/f'{split}-repair-prompts.json')
        assert set(done)<=set(old) and set(attempts)<={r['id'] for r in prompts}
        for r in list(done.values())+list(attempts.values()):validate(r['run_binding'],b)
        def commit(key,candidate=None):
            issues=[];elapsed=0.
            if candidate is not None:
                begun=time.perf_counter();issues=inspect_sql(candidate['pred_sql'],checkmeta[old[key]['db_id']]);elapsed=time.perf_counter()-begun
            result=finalize(old[key],checks[key],candidate,issues,elapsed)
            result['run_binding']=b;put(path,result);done[key]=result
        for key,check in checks.items():
            if key not in done and not check['eligible']:commit(key)
        # Recover durable attempts before making any GPU call.
        for key,candidate in attempts.items():
            if key not in done:commit(key,candidate)
        for pair in pending_batches(prompts,set(done)):
            if model is None:
                pilot.OUT=OUT/'runtime';pilot.OUT.mkdir(parents=True,exist_ok=True)
                model,tok=pilot.load(train=False,adapter=ADAPTER)
            generated=inc.generate(model,tok,pair,meta,'syntax-binding')
            assert {r['id'] for r in generated}=={r['id'] for r in pair}
            for candidate in generated:
                candidate.update(run_binding=b,group='corrected',adapter=str(ADAPTER),decode='syntax-binding')
                put(attemptpath,candidate);attempts[candidate['id']]=candidate
            for candidate in generated:commit(candidate['id'],candidate)
            if not (OUT/'startup-ready.json').exists():
                save(OUT/'startup-ready.json',dict(passed=True,split=split,ids=[r['id'] for r in pair],
                    statuses=[r['generation_status'] for r in generated],finite_raw_logits=True,batch_limit=2))
            save(OUT/'progress.json',dict(split=split,completed=len(done),total=count,attempts=len(attempts)))
            print('REPAIR',split,len(done),count,'attempts',len(attempts),flush=True)
        assert len(done)==count
        export(path)
        if attempts:export(attemptpath)
    save(OUT/'groups/corrected/generation-summary.json',dict(completed=True,binding=b))


def folder(group):
    return EVAL/'D' if group=='D' else OUT/'original' if group=='original' else OUT/'groups/corrected'


def score():
    import combined_ablation_score as s
    # Copy only existing reference outcomes; no old SQL is re-executed.
    shutil.copytree(OLD/'reference-results',OUT/'reference-results',dirs_exist_ok=True)
    with patch.multiple(s,OUT=OUT,GROUPS=['D','original','corrected'],folder=folder,infer_binding=lambda group:binding(group)):
        s.score('corrected')


def report():
    from generalization_runtime import summarize
    from combined_ablation_score import paired
    groups={};contrasts={};recovery={};correction={}
    lines=['# 旧去掉DPO＋静态检查＋一次纠错','','同一adapter，第二次生成仍使用语法＋绑定约束；不训练。']
    for split,count in SPLITS.items():
        scores={k:records(folder(k)/f'{split}-scores.jsonl') for k in ['original','corrected']}
        ids={r['id'] for r in read(OUT/f'{split}-prompts.json')}
        assert all(len(v)==count and {r['id'] for r in v}==ids for v in scores.values())
        groups[split]={k:summarize(v) for k,v in scores.items()}
        contrasts[split]=paired(scores['original'],scores['corrected'],split=='main')
        baseline={r['id']:r for r in records(EVAL/'baseline'/f'{split}-scores.jsonl')}
        d={r['id']:r for r in records(EVAL/'D'/f'{split}-scores.jsonl')}
        lost=sorted(i for i in ids if baseline[i]['repository_ex'] and not d[i]['repository_ex'])
        if split=='financial':assert len(lost)==18
        recovery[split]={}
        for k,rr in scores.items():
            sm={r['id']:r for r in rr}
            recovery[split][k]=dict(original_losses=lost,recovered=[i for i in lost if sm[i]['repository_ex']],
                new_regressions=sorted(i for i in ids if d[i]['repository_ex'] and not sm[i]['repository_ex']))
        pp=records(folder('corrected')/f'{split}-predictions.jsonl');checks=read(OUT/f'{split}-checks.json')
        attempts=records(folder('corrected')/f'{split}-attempts.jsonl')
        correction[split]=dict(flagged=sum(r['flagged'] for r in checks),eligible=sum(r['eligible'] for r in checks),
            attempted=sum(r['correction_attempted'] for r in pp),accepted=sum(r['correction_accepted'] for r in pp),
            rejected=sum(r['correction_attempted'] and not r['correction_accepted'] for r in pp),
            changed=sum(r['sql_changed'] for r in pp),repeated=sum(r['repeated_sql'] for r in pp),
            original_generation_seconds=sum(r['seconds'] for r in pp),
            extra_generation_seconds=sum(r['extra_generation_seconds'] for r in pp),
            extra_generated_tokens=sum(r.get('extra_generated_tokens',0) for r in pp),
            extra_parser_seconds=sum(r['extra_parser_seconds'] for r in pp),
            static_seconds=sum(r['seconds'] for r in checks)+sum(r['postcheck_seconds'] for r in pp),
            attempt_statuses=dict(Counter(r['generation_status'] for r in attempts)))
        lines+=['','## '+split,'','|组|EX|可执行率|set EX|bag EX|','|---|---:|---:|---:|---:|']
        for k,r in groups[split].items():lines.append(f"|{k}|{r['ex']:.1%}|{r['valid']:.1%}|{r['set_ex']:.1%}|{r['bag_ex']:.1%}|")
        c=contrasts[split]
        lines+=['',f"EX差值 {c['delta']*100:+.1f}pp；修复{len(c['repairs'])}题，退步{len(c['harms'])}题。"+
            (f" 数据库聚类配对95%区间[{c['ci95'][0]*100:+.1f},{c['ci95'][1]*100:+.1f}]pp。" if split=='main' else '')]
        for k,r in recovery[split].items():lines+=[f"{k}：原{len(lost)}题恢复{len(r['recovered'])}，相对D新增退步{len(r['new_regressions'])}。"]
        c=correction[split]
        lines += ['',f"触发{c['flagged']}，可尝试{c['eligible']}，尝试{c['attempted']}，接受{c['accepted']}，拒绝{c['rejected']}，SQL改变{c['changed']}，重复原SQL{c['repeated']}。",
                  f"原生成历史耗时{c['original_generation_seconds']/60:.2f}分钟；额外生成{c['extra_generation_seconds']/60:.2f}分钟，含解析{c['extra_parser_seconds']:.2f}秒；静态检查及诊断{c['static_seconds']:.2f}秒。",
                  '候选状态：'+str(c['attempt_statuses'])]
    save(OUT/'comparison.json',dict(groups=groups,contrasts=contrasts,recovery=recovery,correction=correction,config=CONFIG))
    lines += ['','主100题13道超长题和所有失败保留分母。只读评分30秒、100000行、64MiB；不根据参考答案接受候选。',
              '固定规则不按结果修改，单种子、已查看开发集；可执行率提高不能单独证明SQL语义或通用能力提升。',
              '额外解析时间包含在额外生成时间内，不重复相加；模型加载与评分另见stage-costs。无哈希核对、额外复核或旧模型重放。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def pipeline():
    prepare()
    for stage in ['infer','score','report']:
        key=str(time.time_ns());log=OUT/'logs'/(key+'-'+stage+'.log');log.parent.mkdir(exist_ok=True)
        save(OUT/'status.json',dict(status='running',stage=stage,log=str(log)))
        started=time.time()
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),stage],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(OUT/'stage-costs'/(key+'.json'),dict(stage=stage,seconds=time.time()-started,returncode=result.returncode))
        if result.returncode:
            save(OUT/'status.json',dict(status='failed',stage=stage,log=str(log)));raise SystemExit(result.returncode)
    save(OUT/'status.json',dict(status='completed',report=str(OUT/'REPORT.md')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','score','report','pipeline']);args=p.parse_args()
    globals()[args.stage]()
