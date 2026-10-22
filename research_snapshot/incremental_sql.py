"""D inference with incremental SQL constraints. No training, hashes, or review stage."""
import argparse,json,os,shutil,sys,time,subprocess
from pathlib import Path
from collections import Counter
from pilot import read,save
import pilot
from incremental_constraints import ROOT,OUT,from_tokenizer,make_matcher,grammar_text,BindingGuard,LexicalGuard,lexical_grammar
from structure_dpo_score import records,put,export
EVAL=ROOT/'experiments/hardpair-v1/inference-batch2-v1'
D=ROOT/'experiments/data-mixture-v1/D/adapter'
MODES=['syntax','syntax-binding']
ENGINE_CACHE={}
CONFIG=dict(batch_size=2,max_new_tokens=512,prompt_limit=1792,do_sample=False,seed=3407,adapter=str(D),
    model='Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    dependency='llguidance==1.9.1',backend='SDPA MATH',precision='existing D NF4/FP16/FP32',
    dead_end='fail, no backtracking or unconstrained fallback',grammar_version=2)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():print('ALREADY_PREPARED');return
    tok=pilot.tokenizer();lt=from_tokenizer(tok);template=make_matcher(lt);lex_template=make_matcher(lt,lexical_grammar(grammar_text()))
    train=read(ROOT/'experiments/data-mixture-v1/frozen/D/train.json');coverage=[]
    for r in train:
        m=template.deep_copy();ids=lt.tokenize_str(r['sql']);n=m.try_consume_tokens(ids)
        lexical_error=LexicalGuard(lt,lex_template).check(r['sql'].encode(),True)
        coverage.append(dict(id=r['id'],accepted=n==len(ids) and m.is_accepting() and not lexical_error,lexical_error=lexical_error,accepted_tokens=n,total_tokens=len(ids),
                             stop_text=lt.decode_str(ids[max(0,n-3):n+3]) if n<len(ids) else ''))
    save(OUT/'training-grammar-coverage.json',coverage)
    failures=[r for r in coverage if not r['accepted']]
    print('TRAIN_GRAMMAR',len(train)-len(failures),'/',len(train),failures[:12],flush=True)
    if failures:raise RuntimeError('Training grammar coverage incomplete; resolve before freezing / evaluation')
    for split in ['main','financial']:save(OUT/f'{split}-prompts.json',read(EVAL/f'{split}-prompts.json'))
    save(OUT/'metadata.json',read(ROOT/'experiments/static-repair-v1/metadata.json'))
    # Profile uses only training prompts and training schema. No evaluation answer access.
    # Existing normalized training metadata is converted without reading database contents.
    tm=read(ROOT/'experiments/data-mixture-v1/database-metadata.json');converted={}
    for db,v in tm.items():
        converted[db]=dict(tables={t:dict(columns={c:cv.get('type','TEXT') if isinstance(cv,dict) else cv for c,cv in tv['columns'].items()}) for t,tv in v['tables'].items()})
    save(OUT/'training-metadata.json',converted)
    samples=train[:2]+sorted(train,key=lambda r:len(r['prompt_ids']),reverse=True)[:2]
    save(OUT/'profile-prompts.json',[dict(r,context_covered=True) for r in samples])
    save(OUT/'config.json',CONFIG)
    (OUT/'grammar.lark').write_text(grammar_text(),'utf-8')
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['incremental_sql.py','incremental_constraints.py','incremental_sql.lark','test_incremental_sql.py']:shutil.copyfile(ROOT/name,code/name)
    save(OUT/'frozen.json',dict(prepared=True,training_grammar_accepted=len(train),main=100,financial=106,
         sources=dict(prompts=str(EVAL),adapter=str(D)),no_hashes=True,no_extra_review=True))

class Decoder:
    def __init__(self,tok,n_vocab,mode,rows,metas,grammar=None):
        import numpy as np
        self.np=np;self.tok=tok;self.mode=mode
        key=(id(tok),n_vocab,grammar)
        if key not in ENGINE_CACHE:
            lt=from_tokenizer(tok,n_vocab=n_vocab)
            source=grammar or (OUT/'grammar.lark').read_text('utf-8')
            ENGINE_CACHE[key]=(lt,make_matcher(lt,source),make_matcher(lt,lexical_grammar(source)))
        self.lt,self.template,lex_template=ENGINE_CACHE[key]
        self.states=[]
        for row in rows:
            self.states.append(dict(matcher=self.template.deep_copy(),guard=BindingGuard(metas[row['db_id']]),lexical=LexicalGuard(self.lt,lex_template),raw=b'',
              done=False,failed=False,events=[],tokens=[],parse_seconds=0.,binding_seconds=0.,interventions=0,
              rejected_candidates=0,first_intervention=None,status='ok'))
    def select(self,state,values,eos):
        np=self.np;started=time.perf_counter();m=state['matcher']
        mask=np.frombuffer(m.compute_logit_bias(),dtype=np.uint8)>0
        if m.is_error():raise RuntimeError(m.get_error())
        allowed=np.flatnonzero(mask[:len(values)])
        if not len(allowed):return None,[], 'constraint_dead_end'
        raw_top=int(np.argmax(values));reject=[]
        # Exact score order over all legal candidates, deterministic smallest-ID ties.
        ranked=allowed[np.lexsort((allowed,-values[allowed]))]
        chosen=None
        for candidate in ranked:
            candidate=int(candidate)
            chunk=b'' if candidate==eos else self.lt.decode_bytes([candidate])
            raw=state['raw']+chunk
            # Only EOS is unambiguous. Semicolons/backticks may be inside an
            # unfinished SQL string, or be the opening Markdown fence.
            final=candidate==eos
            reason=state['lexical'].check(raw,final)
            if not reason and self.mode=='syntax-binding':
                binding_start=time.perf_counter();reason=state['guard'].check(raw,final)
                state['binding_seconds']+=time.perf_counter()-binding_start
            if reason:reject.append(dict(token=candidate,reason=reason));continue
            chosen=candidate;break
        state['parse_seconds']+=time.perf_counter()-started
        state['rejected_candidates']+=len(reject)
        if chosen!=raw_top:
            state['interventions']+=1
            if state['first_intervention'] is None:state['first_intervention']=len(state['tokens'])
            state['events'].append(dict(position=len(state['tokens']),original_top=raw_top,chosen=chosen,
              original_top_reason='syntax' if not mask[raw_top] else next((r['reason'] for r in reject if r['token']==raw_top),'constraint'),rejected=reject))
        return chosen,reject,'ok' if chosen is not None else 'constraint_dead_end'
    def __call__(self,input_ids,scores):
        import torch
        if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Non-finite raw logits before constraint')
        cpu=scores.detach().float().cpu().numpy().copy();eos=self.tok.eos_token_id
        for j,state in enumerate(self.states):
            scores[j].fill_(float('-inf'))
            if state['done']:scores[j,eos]=0;continue
            chosen,rejected,status=self.select(state,cpu[j],eos)
            if chosen is None:
                state.update(done=True,failed=True,status=status);scores[j,eos]=0;continue
            if not state['matcher'].consume_token(chosen):raise RuntimeError(state['matcher'].get_error())
            state['tokens'].append(chosen)
            if chosen==eos:state['done']=True
            else:state['raw']+=self.lt.decode_bytes([chosen])
            scores[j,chosen]=0 # greedy only: preserve exactly the highest-scoring admissible choice
        return scores

def generate(model,tok,rows,metas,mode):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers import LogitsProcessorList
    from hardpair_batch2 import pad_inputs
    from experiment import extract_sql
    proc=Decoder(tok,model.config.vocab_size,mode,rows,metas);ids,mask=pad_inputs(rows,tok.pad_token_id,'cuda')
    torch.cuda.synchronize();start=time.perf_counter();model.eval()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=512,use_cache=True,
            logits_processor=LogitsProcessorList([proc]),pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;result=[]
    for row,s in zip(rows,proc.states):
        raw=s['raw'].decode('utf-8',errors='replace');complete=s['done'] and not s['failed']
        result.append(dict(id=row['id'],db_id=row['db_id'],mode=mode,raw_output=raw,pred_sql=extract_sql(raw),
            token_ids=s['tokens'],generated_tokens=len(s['tokens']),prompt_tokens=len(row['prompt_ids']),
            completed=complete,generation_status=s['status'] if s['done'] else 'output_incomplete',
            seconds=elapsed/len(rows),batch_seconds=elapsed,batch_ids=[r['id'] for r in rows],actual_batch_size=len(rows),
            parser_seconds=s['parse_seconds'],binding_seconds=s['binding_seconds'],interventions=s['interventions'],
            rejected_candidates=s['rejected_candidates'],first_intervention=s['first_intervention'],events=s['events']))
    return result

def load(mode):
    pilot.OUT=OUT/mode/'runtime';return pilot.load(train=False,adapter=D)

def profile(mode):
    import torch
    model,tok=load('profile-'+mode);rows=read(OUT/'profile-prompts.json');meta=read(OUT/'training-metadata.json');results=[]
    for i in range(0,len(rows),2):results+=generate(model,tok,rows[i:i+2],meta,mode)
    save(OUT/f'profile-{mode}.json',dict(passed=True,rows=results,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
    print('PROFILE',mode,[(r['id'],r['generation_status'],r['interventions']) for r in results],flush=True)

def infer(mode):
    import torch
    from hardpair_batch2 import batches
    assert read(OUT/f'profile-{mode}.json')['passed'];model,tok=load(mode);meta=read(OUT/'metadata.json');costs={}
    for split,count in [('main',100),('financial',106)]:
        rows=read(OUT/f'{split}-prompts.json');path=OUT/mode/f'{split}-predictions.jsonl';done={r['id']:r for r in records(path)}
        for r in rows:
            if not r['context_covered'] and r['id'] not in done:
                item=dict(id=r['id'],db_id=r['db_id'],mode=mode,raw_output='',pred_sql='',token_ids=[],generated_tokens=0,
                    prompt_tokens=len(r['prompt_ids']),completed=False,generation_status='context_limit',seconds=0.,events=[],interventions=0,parser_seconds=0.,binding_seconds=0.,rejected_candidates=0)
                put(path,item);done[r['id']]=item
        for pair in batches(rows):
            if all(r['id'] in done for r in pair):continue
            for item in generate(model,tok,pair,meta,mode):
                if item['id'] not in done:put(path,item);done[item['id']]=item
            save(OUT/'progress.json',dict(mode=mode,split=split,completed=len(done),total=count))
            print('INFER',mode,split,len(done),count,flush=True)
        assert len(done)==count;export(path)
        costs[split]=dict(seconds=sum(r['seconds'] for r in done.values()),generated_tokens=sum(r['generated_tokens'] for r in done.values()),
            parser_seconds=sum(r['parser_seconds'] for r in done.values()),binding_seconds=sum(r['binding_seconds'] for r in done.values()),
            statuses=dict(Counter(r['generation_status'] for r in done.values())))
    save(OUT/mode/'generation-summary.json',dict(completed=True,splits=costs,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))

def score(mode):
    import structure_dpo_score as s
    # Reuse available primary reference-result caches rather than replaying them.
    for parent in [ROOT/'experiments/structure-dpo-v1/reference-results',ROOT/'experiments/D-kl-v1/reference-results']:
        if parent.exists():
            for src in parent.glob('*.json'):
                dst=OUT/'reference-results'/src.name
                if not dst.exists():dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
    s.OUT=OUT;s.GROUPS=MODES;s.score(mode)
    for split in ['main','financial']:
        preds={r['id']:r for r in records(OUT/mode/f'{split}-predictions.jsonl')};path=OUT/mode/f'{split}-scores.jsonl'
        for r in records(path):
            p=preds[r['id']]
            if p['generation_status'] not in ['ok','context_limit','output_incomplete']:
                r['status']=p['generation_status'];put(path,r)
        export(path)

def report():
    from generalization_runtime import summarize,contrast
    result={};contrasts={};recovery={};costs={}
    for split in ['main','financial']:
        scores={g:records((EVAL/g if g in ['baseline','D'] else OUT/g)/f'{split}-scores.jsonl') for g in ['baseline','D']+MODES}
        result[split]={g:summarize(rr) for g,rr in scores.items()}
        contrasts[split]={b+'-'+a:contrast(scores[a],scores[b]) for a,b in [('D','syntax'),('D','syntax-binding'),('syntax','syntax-binding')]}
        maps={g:{r['id']:r for r in rr} for g,rr in scores.items()}
        lost=[i for i,r in maps['baseline'].items() if r['repository_ex'] and not maps['D'][i]['repository_ex']]
        recovery[split]={g:dict(original_losses=lost,recovered=[i for i in lost if maps[g][i]['repository_ex']]) for g in MODES}
        costs[split]={}
        for g in ['D']+MODES:
            pp=records((EVAL/g if g=='D' else OUT/g)/f'{split}-predictions.jsonl');total=sum(r['seconds'] for r in pp);nt=sum(r['generated_tokens'] for r in pp)
            costs[split][g]=dict(seconds=total,tokens=nt,tokens_per_second=nt/total,parser_seconds=sum(r.get('parser_seconds',0) for r in pp),
                intervention_questions=sum(r.get('interventions',0)>0 for r in pp),interventions=sum(r.get('interventions',0) for r in pp),
                intervention_reasons=dict(Counter(e['original_top_reason'].split(':')[0] for r in pp for e in r.get('events',[]))),
                errors=dict(Counter(r.get('error','').split(':')[0] for r in scores[g] if r.get('error'))))
    discarded=[r for split in ['main','financial'] for r in records(OUT/'syntax-before-lexical-fix'/f'{split}-predictions.jsonl')]
    archive_cost=dict(recorded_questions=len(discarded),recorded_generation_seconds=sum(r['seconds'] for r in discarded),
        note='Excluded pre-fix attempt; lower bound on extra cost, excludes interrupted unsaved batch and setup.')
    save(OUT/'comparison.json',dict(groups=result,contrasts=contrasts,recoveries=recovery,costs=costs,discarded_attempt_cost=archive_cost,
        generation_summaries={g:read(OUT/g/'generation-summary.json') for g in MODES},stage_costs=[read(p) for p in sorted((OUT/'stage-costs').glob('*.json'))]))
    lines=['# D 增量 SQL 解析三组对照','','固定 D adapter；不训练，BIRD 主100题、financial106题，batch2、greedy512。D复用历史输出，两新增组无回退、无事后纠错。']
    for split in ['main','financial']:
        lines += ['',f'## {split}','','|组别|EX|可执行率|set EX|bag EX|','|---|---:|---:|---:|---:|']
        for g in ['D']+MODES:
            s=result[split][g];lines.append(f"|{g}|{s['ex']:.1%}|{s['valid']:.1%}|{s['set_ex']:.1%}|{s['bag_ex']:.1%}|")
        for name,c in contrasts[split].items():
            lines+=['',f"{name}: {c['delta']*100:+.1f}pp；修复{len(c['repairs'])}题、误改{len(c['harms'])}题。"+(f"数据库聚类bootstrap95%区间[{100*c['ci95'][0]:+.1f},{100*c['ci95'][1]:+.1f}]pp。" if split=='main' else '')]
        if split=='financial':
            for g,r in recovery[split].items():lines+=['',f"{g} 恢复原{len(r['original_losses'])}道退步中的{len(r['recovered'])}道。"]
        lines+=['','|组别|生成秒|token数|token/s|约束CPU秒|介入题数|','|---|---:|---:|---:|---:|---:|']
        for g,c in costs[split].items():lines.append(f"|{g}|{c['seconds']:.2f}|{c['tokens']}|{c['tokens_per_second']:.2f}|{c['parser_seconds']:.2f}|{c['intervention_questions']}|")
        lines+=['','失败状态：'+json.dumps({g:result[split][g]['statuses'] for g in ['D']+MODES},ensure_ascii=False)]
        lines+=['','执行错误类别：'+json.dumps({g:costs[split][g]['errors'] for g in ['D']+MODES},ensure_ascii=False)]
        lines+=['','最高分token被改变的原因：'+json.dumps({g:costs[split][g]['intervention_reasons'] for g in MODES},ensure_ascii=False)]
    lines += ['','## 方法与限制','','llguidance 1.9.1 逐token语法掩码；候选按原模型分数选择。SQLGlot 27.29.0只在可解析表达式/查询处辅助绑定，未完成前缀解析异常不构成拒绝理由。SELECT先于FROM的绑定延迟，已经明确绑定的物理列及时检查；全部来源范围确定或EOS时完成作用域检查。无候选合法则失败，不重生成。',
      '约束组统一只允许原始SQL或SQL代码块；输出格式约束也可能贡献差异。函数名字没有执行白名单，合法语法和绑定不能保证函数存在、JOIN语义、筛选条件正确或查询能在资源限额内执行。',
      '所有失败保留分母，主评测13道超长题未删除。只读SQLite评分采用既有30秒、100000行、64MiB限制。已存在的同题完全相同SQL和完成状态可复用评分，其余做本轮主要评分；没有额外复核或哈希核对。',
      'D耗时是历史运行参考；新增组采用相同计时，解析成本包含于生成时间。计时不含模型加载及语法/tokenizer初始化（按进程缓存），批外初始化另见阶段总时间。',
      f"实现阶段修正了原始CFG不能确保词法边界的问题，增加最大词法单元的影子增量解析，解析器分隔标记不会进入模型文本。修正前部分运行已排除计分；保留{len(discarded)}条记录，其已记录生成时间{archive_cost['recorded_generation_seconds']:.2f}秒作为额外成本下界。没有用评测答案或EX选择修正。两组均使用修正后的实现重新生成。",
      '本轮数据均已查看，属于开发实验。可执行率改善不等于EX改善；不将小样本结果推广为通用能力。逐题修复/误改ID、介入位置和候选拒绝原因见JSON/JSONL。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8');save(OUT/'status.json',dict(status='completed'))
    print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)

def pipeline():
    stages=[['profile','--mode',m] for m in MODES if not (OUT/f'profile-{m}.json').exists()]+[['infer','--mode',m] for m in MODES]+[['score','--mode',m] for m in MODES]+[['report']]
    (OUT/'logs').mkdir(exist_ok=True)
    for args in stages:
        log=OUT/'logs'/('-'.join(args)+'.log');start=time.time();save(OUT/'status.json',dict(status='running',stage=args,log=str(log)))
        with log.open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(OUT/'stage-costs'/('-'.join(args)+'.json'),dict(args=args,seconds=time.time()-start,returncode=r.returncode))
        if r.returncode:save(OUT/'status.json',dict(status='failed',stage=args,log=str(log)));raise SystemExit(r.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','profile','infer','score','report','pipeline']);p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.stage in ['profile','infer','score']:globals()[a.stage](a.mode)
    else:globals()[a.stage]()
