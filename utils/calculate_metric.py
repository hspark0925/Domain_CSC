import jieba
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import ipdb
import re

def get_typos(src_sent: str, trg_sent: str, keyword: str) -> str:
    matcher = SequenceMatcher(None, src_sent, trg_sent)
    typos = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Extend the pair of from trg_tokens that differ from src_tokens.
            if trg_sent[j1:j2] != keyword:
                typos.extend(trg_sent[j1:j2])
    return typos

def get_target_indices(sentence, target_sent_piece) -> list:
    indices = []
    
    for match in re.finditer(re.escape(target_sent_piece), sentence):
        start_index = match.start()
        end_index = match.end()
        indices.append((start_index, end_index))
    if not indices:
        raise ValueError(f"'{target_sent_piece}' not found in: {sentence}")
    return indices

def context_check(trg_sent, prd_sent, target_sent_piece, k=2):
    """
    Return:
        True if the context around the target token matches, False otherwise.
    """
    trg_indices = get_target_indices(trg_sent, target_sent_piece)
    prd_indices = get_target_indices(prd_sent, target_sent_piece)
    
    trg_context = {}
    prd_context = {}
    
    for trg_index in trg_indices:
        trg_context[trg_index] = trg_sent[max(0, trg_index[0] - k): min(len(trg_sent), trg_index[1] + k)]
    
    for prd_index in prd_indices:
        prd_context[prd_index] = prd_sent[max(0, prd_index[0] - k): min(len(prd_sent), prd_index[1] + k)]
    
    for trg_ctx in trg_context.values():
        for prd_ctx in prd_context.values():
            if trg_ctx == prd_ctx:
                return True # Return immediately if a match is found
    
    return False  # Return False if no match found


def get_label(src_sent, trg_sent, prd_sent, keyword, k):
    """
    If any of the typos (keywords) in trg tokens are not found in prd tokens with the right context, return 0, else 1.
    """
    # For each typos
    err_type = set()
    label = 1
    if keyword:
        if keyword not in prd_sent:
            err_type.add("non_keyword")
            label = 0
        else:
            if not context_check(trg_sent, prd_sent, keyword, k):
                err_type.add("context_mismatch")
                label = 0
            
    #typos without keyword
    typos = get_typos(src_sent, trg_sent, keyword)
    for typo in typos:
        if typo not in prd_sent:
            err_type.add("general_csc_err")
            label = 0
        else:
            if not context_check(trg_sent, prd_sent, typo, k):
                err_type.add("context_mismatch")
                label = 0
    return label, list(err_type), typos

def compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, index, metric_mode, k):
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
    pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents, wrong_sents  = [], [], [], [], [], [], [], []
    for s, t, p, keyword, domain, instruction, i in zip(src_sents, trg_sents, prd_sents, keywords, domains, instructions, index):
        if metric_mode == "token":
            label, err_type, typos = get_label(s, t, p, keyword[1], k)
            if not label:
                wrong_sents.append({"len_s_t_p": f"{len(s)} {len(t)} {len(p)}","domain_instruction": f"在{domain}领域，{instruction}","keywords": keyword, "typos": typos, "src": s, "trg": t, "prd": p, "i": i, "err_type": err_type})

        if s != t:
            pos_sents.append(t)
            if label:
                tp_sents.append({"len_s_t_p": f"{len(s)} {len(t)} {len(p)}","domain_instruction": f"在{domain}领域，{instruction}","keywords": keyword, "typos": typos, "src": s, "trg": t, "prd": p, "i": i})
            if p == s:
                fn_sents.append({"len_s_t_p": f"{len(s)} {len(t)} {len(p)}","domain_instruction": f"在{domain}领域，{instruction}", "src": s, "trg": t, "prd": p})
        else: #s == t
            neg_sents.append(t)
            
            # If the keyword in trg tokens is not found in prd tokens with the right context.
            if not label:
                fp_sents.append({"len_s_t_p": f"{len(s)} {len(t)} {len(p)}", "src": s, "trg": t, "prd": p})

        if s != p:
            prd_pos_sents.append(p)
        if s == p:
            prd_neg_sents.append(p)

    # Compute precision, recall, F1 score, and false positive rate
    p = len(tp_sents) / (len(prd_pos_sents) + 1e-12)
    r = len(tp_sents) / (len(pos_sents) + 1e-12)
    f1 = 2.0 * (p * r) / (p + r + 1e-12)
    fpr = (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

    

    return p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wrong_sents