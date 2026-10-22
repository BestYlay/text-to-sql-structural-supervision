"""The historical author comparator is typed-string bag equality, not official BIRD EX."""
import re
from collections import Counter
_FENCE_RE=re.compile(r"```(?:sql)?\s*(.*?)```",re.DOTALL|re.IGNORECASE)

def result_sets_match(first,second):
    return Counter(tuple(str(v) for v in row) for row in first)==Counter(tuple(str(v) for v in row) for row in second)

def extract_sql(text):
    if not text:return ''
    match=_FENCE_RE.search(text)
    candidate=match.group(1) if match else text
    candidate=re.sub(r'^\s*SQL\s*:\s*','',candidate,flags=re.IGNORECASE).strip()
    return candidate.split(';',1)[0].split('\n\n',1)[0].strip()
