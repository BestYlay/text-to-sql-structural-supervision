"""Byte-aware SQL grammar and conservative scope guards; no database execution."""
import re,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'assets'
import llguidance as llg
from llguidance.hf import from_tokenizer
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope,Scope
from sqlglot.optimizer.qualify import qualify

RESERVED=set('SELECT FROM WHERE GROUP HAVING ORDER LIMIT OFFSET JOIN INNER OUTER LEFT RIGHT FULL CROSS NATURAL ON USING AS DISTINCT ALL UNION INTERSECT EXCEPT AND OR NOT IS IN BETWEEN LIKE GLOB MATCH REGEXP ESCAPE NULL TRUE FALSE ISNULL NOTNULL CASE WHEN THEN ELSE END EXISTS CAST WITH RECURSIVE WINDOW OVER PARTITION BY ROWS RANGE GROUPS UNBOUNDED PRECEDING FOLLOWING CURRENT ROW EXCLUDE NO OTHERS TIES FILTER ASC DESC NULLS FIRST LAST COLLATE MATERIALIZED'.split())
# SQLite parse.y permits these keyword tokens to fall back to identifiers.
RESERVED-=set('ASC DESC BY CAST END LIKE GLOB REGEXP MATCH NO OFFSET RECURSIVE ROW ROWS WITH NULLS FIRST LAST CURRENT FOLLOWING PARTITION PRECEDING RANGE UNBOUNDED EXCLUDE GROUPS OTHERS TIES MATERIALIZED INNER OUTER LEFT RIGHT FULL CROSS NATURAL'.split())

def identifier_regex():
    # A trie complement excludes exact reserved words without unsupported lookahead.
    trie={}
    for word in sorted(RESERVED):
        node=trie
        for char in word.lower():node=node.setdefault(char,{})
        node['']=True
    def suffix(node):
        kids=[c for c in node if c];parts=[]
        if '' not in node:parts.append('')
        excluded=''.join(kids)
        charset='[a-z_0-9$\\x{80}-\\x{10ffff}'+('&&[^'+excluded+']' if kids else '')+']'
        parts.append(charset+'[a-z_0-9$\\x{80}-\\x{10ffff}]*')
        parts += [c+suffix(node[c]) for c in kids]
        return '(?:'+'|'.join(parts)+')'
    kids=list(trie);other='[a-z_\\x{80}-\\x{10ffff}&&[^'+''.join(kids)+']][a-z_0-9$\\x{80}-\\x{10ffff}]*'
    return '(?i:'+ '|'.join([other]+[c+suffix(trie[c]) for c in kids])+')'

def grammar_text():return (ROOT/'incremental_sql.lark').read_text('utf-8').replace('__IDENT_PATTERN__',identifier_regex())

def make_matcher(lt,grammar=None):
    m=llg.LLMatcher(lt,grammar or grammar_text(),log_level=0)
    if m.is_error():raise RuntimeError(m.get_error())
    return m

def lexical_grammar(grammar):
    """Atomic lexeme grammar: separator sentinels are parser-only, never model text.

    CFG string terminals alone permit SELECT1 => SELECT + 1. The shadow parser
    receives maximal SQLite lexemes separated by a sentinel, so one identifier
    cannot be split into two grammar symbols. Both parsers are incremental.
    """
    lines=[]
    for line in grammar.splitlines():
        if not line.strip() or line.startswith('%ignore'):continue
        if line.startswith('start:'):line='start: query ";"?'
        if re.match('[A-Z][A-Z_]*:',line):line+=' "\\u001f"'
        else:
            line=re.sub(r'"(?:[^"\\]|\\.)*"',lambda m:json.dumps(json.loads(m.group())+'\x1f'),line)
        lines.append(line)
    return '\n'.join(lines)

def sql_lexemes(text,final=False):
    """Return completed maximal tokens plus a possibly unfinished last lexeme."""
    text=text.lstrip();fenced=text.startswith('```')
    if text in ['`','``'] and not final:return [],'',False,None
    if fenced:
        text=text[3:]
        if len(text)<=3 and 'sql'.startswith(text.lower()):
            return [],'',False,'lexical_incomplete' if final else None
        if text[:3].lower()=='sql':text=text[3:]
    tokens=[];i=0;n=len(text)
    def tail(value):return tokens,value,False,None
    while i<n:
        if text[i].isspace():i+=1;continue
        if fenced and text.startswith('```',i):
            return tokens,'',True,None if not text[i+3:].strip() else 'lexical_after_fence'
        if fenced and text[i:] in ['`','``'] and not final:return tail('')
        if text.startswith('--',i):
            end=text.find('\n',i+2)
            if end<0:return tokens,'',False,None
            i=end+1;continue
        if text.startswith('/*',i):
            end=text.find('*/',i+2)
            if end<0:return tokens,'',False,'lexical_open_comment' if final else None
            i=end+2;continue
        begin=i;c=text[i]
        if c in "'\"`[":
            closing=']' if c=='[' else c;i+=1;closed=False
            while i<n:
                if text[i]==closing:
                    if closing!=']' and i+1<n and text[i+1]==closing:i+=2;continue
                    i+=1;closed=True;break
                i+=1
            if not closed:return tail(text[begin:]) if not final else (tokens,'',False,'lexical_open_quote')
        elif c.isalpha() or c=='_' or ord(c)>=128:
            i+=1
            while i<n and (text[i].isalnum() or text[i] in '_$' or ord(text[i])>=128):i+=1
            # SQLite blob literal: x'0123' is a single token.
            if text[begin:i].lower()=='x' and i<n and text[i]=="'":
                end=text.find("'",i+1)
                if end<0:return tail(text[begin:]) if not final else (tokens,'',False,'lexical_open_blob')
                i=end+1
        elif c.isdigit() or (c=='.' and i+1<n and text[i+1].isdigit()):
            match=re.match(r'(?:0[xX][0-9a-fA-F]*|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]*)?)',text[i:])
            i+=len(match.group())
            if i<n and (text[i].isalpha() or text[i]=='_'):return tokens,'',False,'lexical_numeric_suffix'
        else:
            if not final and text[i:] in ['-','/']:return tail('') # may begin a comment
            op=next((op for op in ['->>','->','<=','>=','<>','!=','==','||','<<','>>'] if text.startswith(op,i)),None)
            i+=len(op) if op else 1
        value=text[begin:i]
        if i==n and not final:return tail(value)
        tokens.append(value)
    return tokens,'',False,None

class LexicalGuard:
    def __init__(self,lt,template):self.lt=lt;self.template=template;self.key=None;self.cached=None;self.results={}
    def check(self,raw,final=False):
        key=(raw,final)
        if key in self.results:return self.results[key]
        try:text=raw.decode('utf-8')
        except UnicodeDecodeError:return 'lexical_utf8' if final else None
        complete,pending,ended,error=sql_lexemes(text,final)
        if error:return error
        prefix=tuple(complete)
        if prefix!=self.key:
            self.key=prefix;self.cached=self.template.deep_copy()
            self.cached.consume_tokens(self.lt.tokenize_str(''.join(v+'\x1f' for v in complete)))
        m=self.cached.deep_copy()
        if m.is_error():result='lexical_token_boundary'
        elif pending and not m.consume_tokens(self.lt.tokenize_str(pending)):result='lexical_token_boundary'
        elif (final or ended) and not m.is_accepting():result='lexical_incomplete'
        else:result=None
        if len(self.results)>10000:self.results.clear()
        self.results[key]=result
        return result

def unwrap(text):
    s=text.lstrip()
    if s.startswith('```'):
        s=s[3:]
        if s.lower().startswith('sql'):s=s[3:]
        elif s and 'sql'.startswith(s.lower()):return ''
        s=s.rstrip()
        if s.endswith('```'):s=s[:-3]
        elif s.endswith('``'):s=s[:-2]
        elif s.endswith('`'):s=s[:-1]
    return s

# This lexer is only for conservative guard scheduling. Grammar legality is LLMatcher's job.
LEX=re.compile(r"--[^\r\n]*|/\*[\s\S]*?\*/|'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`(?:[^`]|``)*`|\[[^\]]*\]|[\w$]+|[^\s]",re.UNICODE)

def scopes_closed(sql,final=False):
    depth=0;scopes=[]
    for m in LEX.finditer(sql):
        t=m.group();u=t.upper()
        if t.startswith(('--','/*',"'",'"','`','[')):continue
        if t=='(':depth+=1
        elif t==')':
            for scope in scopes:
                if scope[0]==depth:scope[1]=True
            depth-=1
        elif u=='SELECT':scopes.append([depth,False])
        elif u in {'WHERE','GROUP','HAVING','WINDOW','ORDER','LIMIT','UNION','INTERSECT','EXCEPT'}:
            for scope in reversed(scopes):
                if scope[0]==depth:scope[1]=True;break
    return bool(scopes) and (final or all(s[1] for s in scopes))

class BindingGuard:
    def __init__(self,metadata):
        self.schema={t.lower():{c.lower():typ for c,typ in v['columns'].items()} for t,v in metadata['tables'].items()}
        self.cache={}
    def check(self,raw,final=False):
        try:text=raw.decode('utf-8') if isinstance(raw,bytes) else raw
        except UnicodeDecodeError:return None # a UTF-8 character split across BPE tokens
        sql=unwrap(text)
        key=(sql,final)
        if key in self.cache:return self.cache[key]
        result=self._check(sql,final)
        if len(self.cache)>10000:self.cache.clear()
        self.cache[key]=result
        return result
    def _check(self,sql,final):
        if not sql.strip():return 'binding_incomplete' if final else None
        # An open lexical item may grow into a different valid identifier.
        if not final and (sql[-1].isalnum() or sql[-1] in '_$'):return None
        try:tree=sqlglot.parse_one(sql,read='sqlite')
        except Exception:return 'binding_unsupported_parse' if final else None
        if not isinstance(tree,exp.Query):return 'binding_unsupported_query' if final else None
        closed=scopes_closed(sql,final)
        # Never globally resolve SELECT-before-FROM or an unfinished CTE body.
        # Partial qualified references can be checked once a local alias is bound.
        try:
            for scope in traverse_scope(tree):
                local={a.lower():s for a,s in scope.sources.items()}
                for alias,source in local.items():
                    if isinstance(source,exp.Table) and source.name.lower() not in self.schema:
                        if closed:return 'missing_table:'+source.name
                for col in scope.columns:
                    if not col.table or col.name=='*':continue
                    source=local.get(col.table.lower())
                    if isinstance(source,exp.Table) and source.name.lower() in self.schema:
                        if col.name.lower() not in self.schema[source.name.lower()] and col.name.lower() not in {'rowid','oid','_rowid_'}:
                            return 'missing_column:'+source.name+'.'+col.name
            if closed:
                # Infer no missing columns/tables. Qualify resolves CTE/derived aliases
                # and correlated scopes, including select-list aliases in SQLite.
                normalized=tree.copy()
                # SQLite names remain case insensitive inside identifier quotes.
                for name in normalized.find_all(exp.Identifier):name.set('this',name.this.lower())
                qualify(normalized,dialect='sqlite',schema=self.schema,infer_schema=False,validate_qualify_columns=True)
        except sqlglot.errors.OptimizeError as e:return 'binding:'+str(e)
        except Exception as e:return 'binding_unsupported:'+type(e).__name__ if final else None
        return None
