from typing import Dict, Tuple, List
import numpy as np

# Minimal soft constraints and reranking utilities
# Define some known symmetric, inverse, and antisymmetric relations using Wikidata PIDs
SYMMETRIC = {
    'P26',   # spouse
    'P3373', # sibling
    'P190',  # sister city
}

INVERSES: Dict[str, str] = {
    'P1376': 'P36',  # capital of  <-> capital
    'P36': 'P1376',
    'P527': 'P361',  # has part <-> part of
    'P361': 'P527',
    'P155': 'P156',  # follows <-> followed by
    'P156': 'P155',
    'P150': 'P131',  # contains admin territory <-> located in admin territory
    'P131': 'P150',
    'P40':  'P22',   # child <-> father (approx; also P25 mother)
    'P22':  'P40',
}
# Antisymmetric-like pairs (keep one direction)
ANTISYMMETRIC = {
    'P22', # father
    'P25', # mother
    'P40', # child (inverse direction)
}

def apply_constraints(doc_triples: List[Tuple[int,int,str,float]]) -> List[Tuple[int,int,str,float]]:
    """Given a list of (h,t,r,score) for a single document, apply simple constraints:
       - Make symmetric relations bi-directional keeping min score
       - For inverse pairs, if only one exists, add the inverse with same score
       - For antisymmetric ones, if both (h,t,r) and (t,h,r) present, keep higher-score one
    Returns a possibly modified list.
    """
    # Index for quick lookup
    by_key: Dict[Tuple[int,int,str], float] = {}
    for h,t,r,s in doc_triples:
        by_key[(h,t,r)] = max(s, by_key.get((h,t,r), -1e9))

    # Symmetric enforcement
    for (h,t,r), s in list(by_key.items()):
        if r in SYMMETRIC and h != t:
            pair = (t,h,r)
            if pair not in by_key:
                by_key[pair] = s
            else:
                # unify scores conservatively
                by_key[pair] = max(by_key[pair], s)

    # Inverses propagation
    for (h,t,r), s in list(by_key.items()):
        inv = INVERSES.get(r)
        if inv is not None and h != t:
            inv_key = (t,h,inv)
            if inv_key not in by_key:
                by_key[inv_key] = s
            else:
                by_key[inv_key] = max(by_key[inv_key], s)

    # Antisymmetric pruning
    for r in ANTISYMMETRIC:
        keys = [(h,t,k) for (h,t,k) in list(by_key.keys()) if k == r]
        for (h,t,_r) in keys:
            if h == t:
                continue
            a = (h,t,r)
            b = (t,h,r)
            if a in by_key and b in by_key:
                if by_key[a] >= by_key[b]:
                    by_key.pop(b, None)
                else:
                    by_key.pop(a, None)

    # Return sorted by score desc
    items = [ (h,t,r,s) for (h,t,r), s in by_key.items() ]
    items.sort(key=lambda x: x[3], reverse=True)
    return items

