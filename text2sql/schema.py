import random
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope,traverse_scope

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
