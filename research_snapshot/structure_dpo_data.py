"""Training-only AST labels and existing execution-distinguished preference pairs."""
import random
from collections import Counter
from pathlib import Path
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope,traverse_scope
import pilot
from pilot import read,save

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/structure-dpo-v1'
SOURCE=ROOT/'experiments/data-mixture-v1'
EVAL=ROOT/'experiments/hardpair-v1/inference-batch2-v1'
CONFIG=dict(seed=3407,model='Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    structure=dict(samples=800,steps=50,accumulation=16,lr=2e-4,warmup=2,alpha=.1,beta=.1,head_dim=128,
        column_table_loss='0.5 balanced BCE tables + 0.5 balanced BCE columns',edge_loss='balanced BCE, all positives + up to 32 deterministic sampled negatives',
        representation='last prompt token final hidden state; mean frozen input embeddings of schema candidate names/types',
        output='unchanged SQL; auxiliary heads discarded for generation',unresolved='keep SQL SFT; mask only unresolved auxiliary labels'),
    preference=dict(max_pairs=192,min_pairs=160,accumulation=8,lr=5e-6,warmup=1,beta=.1,epochs=1,
        initialization='D adapter',reference='frozen D adapter, precomputed completion log probability sums',
        loss='original sigmoid DPO; summed completion log probabilities, no length normalization',control='chosen-only SFT from D, same pairs/order/steps/lr'),
    shared=dict(rank=16,alpha=16,dropout=0,optimizer='AdamW8bit',weight_decay=.01,clip=1.,scale=256,
        checkpointing='non-reentrant',precision='NF4 double, FP16 compute, FP32 adapter/nonquantized',backend='SDPA MATH',
        prompt_limit=1792,sequence_limit=2048,batch_size=2,max_new_tokens=512,do_sample=False,
        evaluation='BIRD main100/8 databases and financial106, existing viewed development splits',correction=False),
    hash_checks=False,independent_review=False)

def ast_labels(sql,metadata):
    schema={t:{c:v['type'] for c,v in item['columns'].items()} for t,item in metadata['tables'].items()}
    tree=qualify(sqlglot.parse_one(sql,read='sqlite'),dialect='sqlite',schema=schema,infer_schema=False,
                 validate_qualify_columns=True,quote_identifiers=True)
    scopes=list(traverse_scope(tree));owners={};used_tables=set()
    for s in scopes:
        for c in s.columns:owners.setdefault(id(c),s)
        for src in s.sources.values():
            if isinstance(src,exp.Table):
                name=src.name.lower()
                if name not in schema:raise ValueError('Unknown physical table '+name)
                used_tables.add(name)
    def resolve(column,scope,seen=frozenset()):
        key=(id(scope),column.table.lower(),column.name.lower())
        if key in seen:raise ValueError('Cyclic lineage')
        seen=seen|{key};alias=column.table.lower();name=column.name.lower()
        current=scope;src=None
        while current is not None:
            sources={a.lower():v for a,v in current.sources.items()}
            if alias in sources:src=sources[alias];break
            current=current.parent
        if src is None:raise ValueError('Unresolved column '+column.sql())
        if isinstance(src,exp.Table):
            t=src.name.lower()
            if name not in schema[t]:raise ValueError('Unknown column '+t+'.'+name)
            return {(t,name)}
        if not isinstance(src,Scope):raise ValueError('Unsupported source')
        if not isinstance(src.expression,exp.Select):raise ValueError('Set-operation lineage requires explicit handling')
        selected=[s for s in src.expression.selects if s.alias_or_name.lower()==name]
        if len(selected)!=1:raise ValueError('Ambiguous derived output '+name)
        answer=set()
        for c in selected[0].find_all(exp.Column):
            answer.update(resolve(c,owners.get(id(c),src),seen))
        return answer
    columns=set()
    for s in scopes:
        for c in s.columns:columns.update(resolve(c,owners.get(id(c),s)))
    edges=set()
    for eq in tree.find_all(exp.EQ):
        if not isinstance(eq.left,exp.Column) or not isinstance(eq.right,exp.Column):continue
        l=resolve(eq.left,owners[id(eq.left)]);r=resolve(eq.right,owners[id(eq.right)])
        # Association labels are physical column equality edges, including WHERE joins.
        # Same-table different aliases count; this head does not encode alias roles.
        if len(l)==len(r)==1:
            a=next(iter(l));b=next(iter(r))
            if a[0]!=b[0] or eq.left.table.lower()!=eq.right.table.lower():edges.add(tuple(sorted((a,b))))
    return dict(tables=sorted(used_tables),columns=sorted(columns),edges=sorted(edges))

def candidates(metadata,tok):
    tables=sorted(metadata['tables']);columns=sorted((t,c) for t in tables for c in metadata['tables'][t]['columns'])
    texts=['TABLE "'+metadata['tables'][t]['name']+'"' for t in tables]
    texts += ['COLUMN "'+metadata['tables'][t]['name']+'"."'+metadata['tables'][t]['columns'][c]['name']+'" '+metadata['tables'][t]['columns'][c]['type'] for t,c in columns]
    return dict(tables=tables,columns=columns,candidate_ids=[tok(x,add_special_tokens=False)['input_ids'] for x in texts])

def labels_for(sql,metadata,candidate,seed):
    raw=ast_labels(sql,metadata);tabs=candidate['tables'];cols=[tuple(x) for x in candidate['columns']];ci={c:i for i,c in enumerate(cols)}
    positives=sorted({tuple(sorted((ci[tuple(a)],ci[tuple(b)]))) for a,b in raw['edges']})
    rng=random.Random(seed);negative=set();attempts=0
    while len(negative)<32 and attempts<2048 and len(cols)>1:
        attempts+=1;a,b=sorted(rng.sample(range(len(cols)),2))
        if cols[a][0]==cols[b][0] or (a,b) in positives:continue
        negative.add((a,b))
    edge_items=positives+sorted(negative)
    return dict(raw=raw,table_labels=[float(t in raw['tables']) for t in tabs],
        column_labels=[float(c in raw['columns']) for c in cols],edge_items=edge_items,edge_labels=[1.]*len(positives)+[0.]*len(negative))

def completion(prompt_ids,sql,tok):
    # Both alternatives share formatting, alias canonicalization and quoting.
    text='```sql\n'+sql+'\n```'+tok.eos_token+'\n'
    answer=tok(text,add_special_tokens=False)['input_ids'];ids=prompt_ids+answer
    if len(ids)>2048:raise ValueError('sequence length '+str(len(ids)))
    assert tok.eos_token_id in answer
    return dict(input_ids=ids,labels=[-100]*len(prompt_ids)+answer,prompt_ids=prompt_ids,sql=sql)

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data-summary.json').exists():assert read(OUT/'config.json')==CONFIG;return
    tok=pilot.tokenizer();train=read(SOURCE/'frozen/D/train.json');metadata=read(SOURCE/'database-metadata.json')
    assert len(train)==800;schemas={db:candidates(metadata[db],tok) for db in sorted({r['db_id'] for r in train})}
    extra=[];exclusions=[]
    for index,r in enumerate(train):
        try:extra.append(dict(id=r['id'],enabled=True,**labels_for(r['sql'],metadata[r['db_id']],schemas[r['db_id']],3407+index)))
        except Exception as e:extra.append(dict(id=r['id'],enabled=False));exclusions.append(dict(id=r['id'],reason=repr(e)))
    assert sum(r['enabled'] for r in extra)>=640,'Too few reliable AST labels'
    pool={r['id']:r for r in read(SOURCE/'pool-candidates.json')};pp={r['target']:r for r in read(SOURCE/'pair-execution.json')}
    prefs=[];excluded=[]
    for r in train:
        p=pp.get(r['id'])
        if p is None:continue
        if not p['execution_gate'] or p['chosen']['result']['status']!='ok':excluded.append(dict(id=r['id'],reason='negative_not_executable'));continue
        assert p['chosen']['distinguishes'] and p['gold']['status']=='ok' and pool[r['id']]['sql']==r['sql']
        # Reuse accepted execution evidence; do not re-run SQL or compare hashes.
        positive=pool[r['id']]['qualified_sql'];negative=p['chosen']['sql']
        try:
            chosen=completion(r['prompt_ids'],positive,tok);rejected=completion(r['prompt_ids'],negative,tok)
        except ValueError as e:excluded.append(dict(id=r['id'],reason=str(e)));continue
        assert positive!=negative
        prefs.append(dict(id=r['id'],db_id=r['db_id'],tag=p['tag'],chosen=chosen,rejected=rejected,
            evidence='existing pair-execution.json execution_gate and distinguishes; original result accepted without replay',intervention=p['chosen']['intervention']))
    keep=len(prefs)//8*8;excluded += [dict(id=r['id'],reason='tail_to_multiple_of_8') for r in prefs[keep:]];prefs=prefs[:keep]
    assert len(prefs)>=160
    for split,count in [('main',100),('financial',106)]:
        prompts=read(EVAL/f'{split}-prompts.json');assert len(prompts)==count
        assert not {r['db_id'] for r in train}&{r['db_id'] for r in prompts}
        save(OUT/f'{split}-prompts.json',prompts)
    save(OUT/'config.json',CONFIG);save(OUT/'train.json',train);save(OUT/'schema-candidates.json',schemas)
    save(OUT/'structure-labels.json',extra);save(OUT/'structure-exclusions.json',exclusions)
    save(OUT/'preferences.json',prefs);save(OUT/'preference-exclusions.json',excluded)
    summary=dict(structure_count=800,structure_labeled=sum(r['enabled'] for r in extra),structure_steps=50,
        preference_pairs=len(prefs),preference_steps=len(prefs)//8,preference_by_tag=dict(Counter(r['tag'] for r in prefs)),
        preference_databases=len({r['db_id'] for r in prefs}),sqlglot=sqlglot.__version__,negative_source='existing 192 executable distinguishable training mutations; no teacher/eval data',
        reference_answer_used_only_for_training=True,hash_checks=False,independent_review=False)
    save(OUT/'data-summary.json',summary);print('PREPARED',summary,flush=True)

if __name__=='__main__':prepare()
