import time
from pathlib import Path
from .constraints import from_tokenizer,make_matcher,BindingGuard,LexicalGuard,lexical_grammar
OUT=Path(__file__).resolve().parent/"assets"
ENGINE_CACHE={}

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
    from .batching import pad_inputs
    from .metrics import extract_sql
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
