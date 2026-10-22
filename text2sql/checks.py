"""Schema-only diagnostics. No reference SQL, scores, values or model calls."""
from collections import deque
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

def fk_path(a,b,foreign_keys):
    queue=deque([(a,[])]);seen={a}
    while queue:
        table,path=queue.popleft()
        if len(path)>=3:continue
        for edge in foreign_keys:
            left,lc,right,rc=edge
            if table not in (left,right):continue
            target=right if table==left else left
            route=path+[edge]
            if target==b:return route
            if target not in seen:seen.add(target);queue.append((target,route))
    return []

def inspect_sql(sql,metadata):
    issues=[]
    def add(code,message,severity='error'):
        item=dict(code=code,severity=severity,message=message)
        if item not in issues:issues.append(item)
    if not sql.strip():return [dict(code='empty',severity='error',message='SQL output is empty.')]
    try:trees=[t for t in sqlglot.parse(sql,read='sqlite') if t is not None]
    except Exception as e:return [dict(code='parse',severity='error',message=str(e)[:450])]
    if len(trees)!=1:return [dict(code='multiple_statements',severity='error',message='Return exactly one read-only SQL query.')]
    tree=trees[0]
    if not isinstance(tree,exp.Query) or any(isinstance(n,(exp.Insert,exp.Update,exp.Delete,exp.Create,exp.Drop,exp.Command)) for n in tree.walk()):
        return [dict(code='nonquery',severity='error',message='Only a read-only SELECT query is allowed.')]
    tables={t.lower():v for t,v in metadata['tables'].items()}
    schema={t:{c:typ for c,typ in v['columns'].items()} for t,v in metadata['tables'].items()}
    try:qualify(tree.copy(),dialect='sqlite',schema=schema,infer_schema=False,validate_qualify_columns=True)
    except Exception as e:add('binding',str(e)[:450])
    # SQLite has no built-in YEAR/MONTH/DAY function. SQLGlot may parse these
    # as dedicated expressions rather than Anonymous nodes.
    for f in tree.find_all(exp.Func):
        name=f.name.upper() if isinstance(f,exp.Anonymous) else f.sql_name().upper()
        if name in {'YEAR','MONTH','DAY'}:add('sqlite_function',f'SQLite has no built-in {name}(); use supported SQLite date functions with appropriate types.')
    for comparison in tree.walk():
        if not isinstance(comparison,(exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE)):continue
        for value,other in [(comparison.left,comparison.right),(comparison.right,comparison.left)]:
            if isinstance(value,exp.TimeToStr) and isinstance(other,exp.Literal) and not other.is_string:
                add('date_type','SQLite STRFTIME returns text. Comparing it directly with a numeric literal can change equality/order semantics; use a compatible quoted date component or an explicit numeric cast.',severity='warning')
    edges=[[str(x).lower() for x in e] for e in metadata['foreign_keys']]
    allowed={(a,c,b,d) for a,c,b,d in edges}|{(b,d,a,c) for a,c,b,d in edges}
    for scope in traverse_scope(tree):
        sources={a.lower():s.name.lower() for a,s in scope.sources.items() if isinstance(s,exp.Table)}
        # Only explicit cross-table equality JOIN predicates; do not label
        # arbitrary non-FK predicates or CTE-derived columns invalid.
        for join in scope.expression.args.get('joins',[]):
            on=join.args.get('on')
            if on is None:continue
            for eq in on.find_all(exp.EQ):
                l,r=eq.left,eq.right
                if not isinstance(l,exp.Column) or not isinstance(r,exp.Column):continue
                a,b=sources.get(l.table.lower()),sources.get(r.table.lower())
                if not a or not b or a==b or a not in tables or b not in tables:continue
                c,d=l.name.lower(),r.name.lower()
                if (a,c,b,d) in allowed:continue
                if not (c in [x.lower() for x in tables[a]['pk']] and d in [x.lower() for x in tables[b]['pk']]):continue
                path=fk_path(a,b,edges)
                if path:
                    route='; '.join(f'{x}.{y} = {z}.{w}' for x,y,z,w in path)
                    add('join_path',f'Check {a}.{c} = {b}.{d}: it is not a declared FK and equates entity keys. Available FK route: {route}. Non-FK joins can be valid; verify against the question.',severity='warning')
    return issues
