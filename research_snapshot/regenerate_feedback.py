"""Schema-only diagnostics; no draft SQL is passed to the fresh prompt."""
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from audit_repair_feedback import routes

def quoted(name):return '"'+name.replace('"','""')+'"'
def field(table,column):return quoted(table)+'.'+quoted(column)

def diagnostics(sql,meta,issues):
    codes={i['code'] for i in issues};lines=[]
    tables={t.lower():(t,{c.lower():c for c in v['columns']}) for t,v in meta['tables'].items()}
    def owners(name):
        return [field(t,cols[name.lower()]) for t,cols in tables.values() if name.lower() in cols]
    try:tree=sqlglot.parse_one(sql,read='sqlite')
    except Exception:tree=None
    if tree is not None:
        for scope in traverse_scope(tree):
            def resolve(alias):
                current=scope
                while current is not None:
                    for key,value in current.sources.items():
                        if key.lower()==alias.lower():return value.name.lower() if isinstance(value,exp.Table) else None
                    current=current.parent
                return None
            for col in scope.columns:
                name=col.name.lower()
                if col.table:
                    table=resolve(col.table)
                    if table in tables and name not in tables[table][1]:
                        lines.append('Invalid field ownership: '+field(tables[table][0],col.name)+' has no column in that table. Existing schema locations with this name: '+(', '.join(owners(name)) or 'none')+'. Choose the intended entity from the question; connect its table using the appropriate relationship.')
                elif 'binding' in codes:
                    locations=owners(name)
                    if len(locations)>1:lines.append('Potentially ambiguous field '+quoted(col.name)+': available at '+', '.join(locations)+'. Qualify the intended table explicitly.')
                    elif not locations:lines.append('No physical schema column named '+quoted(col.name)+'. If this is not a valid derived output alias, use an existing field; do not invent a replacement value.')
            if 'join_path' in codes:
                pairs=set()
                for col in scope.columns:
                    table=resolve(col.table) if col.table else None
                    if table in tables:pairs.add(table)
                edges=[[s.lower() for s in e] for e in meta['foreign_keys']]
                for a in sorted(pairs):
                    for b in sorted(pairs):
                        if a>=b:continue
                        paths=routes(a,b,edges)
                        if paths:
                            lines.append('Candidate declared FK paths between '+quoted(a)+' and '+quoted(b)+' (not semantic recommendations):')
                            for path in paths:lines.append('; '.join(field(x,xc)+' = '+field(y,yc) for x,xc,y,yc in path))
                lines.append('Equating identifiers of different entities is not justified by their similar names. A shared parent or lookup, such as district, only establishes a common attribute; it does not prove ownership or account membership. A bridge table records its declared relationship, but its role/type filters still must follow the question and evidence. Select the relationship by meaning, not the shortest path. Non-FK joins are not automatically invalid.')
    clean={
        'binding':'The schema binding check failed. Verify every field against its owning table and each alias in its own query scope; retain the requested entities and filters.',
        'parse':'The SQL syntax check failed. Generate a complete valid SQLite query. Double-quote identifiers containing spaces or punctuation.',
        'sqlite_function':"SQLite does not provide YEAR/MONTH/DAY functions. Use STRFTIME with the appropriate format. Its result is text: compare to a quoted string or explicitly cast to a number.",
        'date_type':"STRFTIME returns text. Avoid comparing its text output to an unquoted numeric literal; use matching types.",
        'incomplete':'The previous generation did not finish. Produce a concise, complete query without repetition.',
        'empty':'No complete SQL was available. Produce one complete query.',
        'multiple_statements':'Return exactly one SQL query, not multiple statements.',
        'nonquery':'Return only a read-only SELECT query (WITH is allowed).'}
    lines.extend(clean[c] for c in sorted(codes) if c in clean)
    return '\n'.join(dict.fromkeys(lines))

def fresh_messages(original,feedback):
    assert [m['role'] for m in original]==['system','user']
    return [dict(original[0]),dict(role='user',content=original[1]['content']+'\n\nSchema-based diagnostic notes (no reference answer):\n'+feedback+'\n\nGenerate a fresh solution to the original question using its schema and evidence. Preserve all requested filters, output columns and multiplicities. Return only one complete read-only SQLite SQL query in a SQL code block. Do not explain.')]
