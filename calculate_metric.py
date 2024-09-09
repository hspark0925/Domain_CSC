import jieba
import numpy as np
from difflib import SequenceMatcher

def chn_tokenize(sentence):
    return list(jieba.cut(sentence, cut_all=False))

def get_typos_indices(src_tokens, trg_tokens):
    matcher = SequenceMatcher(None, trg_tokens, src_tokens)
    typos = {
        'replace': [],
        'delete': [],
        'insert': []
    }

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Extend the range of indices from trg_tokens that differ from src_tokens.
            typos['replace'].extend(range(i1, i2))
        elif tag == 'delete':
            # Extend the range of indices from trg_tokens that have no matching tokens in src_tokens.
            typos['insert'].extend(range(i1, i2))
        elif tag == 'insert':
            # Extend the range of indices from src_tokens that have no matching tokens in trg_tokens.
            typos['delete'].extend(range(j1, j2))

    return typos

def get_keyword_indices(tokens, keyword):
    typos = {
        'replace': [],
        'delete': [],
        'insert': []
    }
    for i in range(len(tokens)):
        if tokens[i] == keyword:
            typos['replace'].append(i)
    return typos

def get_prd_typo_indices(typo_token, prd_tokens):
    prd_typo_indices = []
    for i in range(len(prd_tokens)):
        if prd_tokens[i] == typo_token:
            prd_typo_indices.append(i)
    return prd_typo_indices

def context_check(trg_tokens, prd_tokens, trg_typo_idex, prd_typo_index, k=3):
    """
    Return:
        True if the context around the typo token matches, False otherwise.
    """
    for offset in range(-k, k + 1):
        trg_context_index = trg_typo_idex + offset
        prd_context_index = prd_typo_index + offset
        
        # Ensure the indices are within the valid range
        if (0 <= trg_context_index < len(trg_tokens)) and (0 <= prd_context_index < len(prd_tokens)):
            if trg_tokens[trg_context_index] != prd_tokens[prd_context_index]:
                return False
        else:
            break
    return True


def get_label(src_tokens, trg_tokens, prd_tokens, typos, k):
    """
    If any of the typos (keywords) in trg tokens are not found in prd tokens with the right context, return 0, else 1.
    """
    # For each typos
    for trg_typo_index in typos['replace'] + typos['insert']:
        trg_typo_token = trg_tokens[trg_typo_index]
        
        prd_typo_indices = get_prd_typo_indices(trg_typo_token, prd_tokens)
        
        context_match = False
        for prd_typo_index in prd_typo_indices:
            # If the context match, the typo is correctly corrected.
            if context_check(trg_tokens, prd_tokens, trg_typo_index, prd_typo_index, k):
                context_match = True
                break
        if not context_match:
            return 0
    
    for src_typo_index in typos['delete']:
        src_typo_token = src_tokens[src_typo_index]
        prd_typo_indices = get_prd_typo_indices(src_typo_token, prd_tokens)
        
        context_match = False
        for prd_typo_index in prd_typo_indices:
            if context_check(src_tokens, prd_tokens, src_typo_index, prd_typo_index, k):
                context_match = True
                break
        if context_match:
            return 0
    return 1

def compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, k):
    """
    tp: src != trg, prd has all the typos with the right context.
    prd_pos: src != prd.
    pos: src != trg.
    fp: src == trg, prd does not have the keyword with the right context.
    neg: src == trg.
    
    p: 모델이 수정한 것 중에서 맞게 수정한 비율  (크면좋음)
    r: 실제로 수정해야 하는 것 중에서 맞게 수정한 비율 (크면좋음)
    fpr: 실제로 수정할 필요가 없는 것 중에서 잘못 수정한 비율 (작으면 좋음)
    """
    pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents = [], [], [], [], [], [], []
    
    for s, t, p, keyword, domain, instruction in zip(src_sents, trg_sents, prd_sents, keywords, domains, instructions):
        src_tokens, trg_tokens, prd_tokens = chn_tokenize(s), chn_tokenize(t), chn_tokenize(p)
        len_s, len_t, len_p = len(src_tokens), len(trg_tokens), len(prd_tokens)
        if s != t:
            pos_sents.append(t)
            
            #If the 
            if get_label(src_tokens, trg_tokens, prd_tokens, get_typos_indices(src_tokens,trg_tokens), k): #p corrected all the (s, t) typos
                tp_sents.append(t)
            if p == s:
                fn_sents.append({"len_s_t_p": f"{len_s} {len_t} {len_p}","domain_instruction": f"在{domain}领域，{instruction}", "src": s, "trg": t, "prd": p})
        else: #s == t
            neg_sents.append(t)
            
            # If the keyword in trg tokens is not found in prd tokens with the right context.
            if not get_label(src_tokens, trg_tokens, prd_tokens, get_keyword_indices(trg_tokens, keyword), k):
                fp_sents.append({"len_s_t_p": f"{len_s} {len_t} {len_p}", "src": s, "trg": t, "prd": p})

        if s != p:
            prd_pos_sents.append(p)
        if s == p:
            prd_neg_sents.append(p)

    # Compute precision, recall, F1 score, and false positive rate
    p = len(tp_sents) / (len(prd_pos_sents) + 1e-12)
    r = len(tp_sents) / (len(pos_sents) + 1e-12)
    f1 = 2.0 * (p * r) / (p + r + 1e-12)
    fpr = (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

    return p, r, f1, fpr, tp_sents, fp_sents, fn_sents