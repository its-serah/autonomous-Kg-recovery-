from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class DocREDDataLoader:
    """Load DocRED documents with text, entities, and relations.
    Produces:
      - docs: dict[doc_id] -> { 'sents': List[List[str]], 'vertexSet': List[List[dict]], 'labels': List[dict] }
      - kg: dict[doc_id] -> List[Tuple[int,int,str]]  (h, t, relation_id)
      - rel2name: dict[str] -> str (e.g., 'P26' -> 'spouse')
    Doc IDs are stable like 'docred_0000'.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.rel_info_path = self.data_dir / 'rel_info.json'
        self.rel2name = self._load_rel_info()

    def _load_rel_info(self) -> Dict[str, str]:
        if self.rel_info_path.exists():
            with open(self.rel_info_path, 'r') as f:
                return json.load(f)
        return {}

    def load_split(self, split: str = 'train_annotated', max_docs: int = None) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[int,int,str]]]]:
        path = self.data_dir / f'{split}.json'
        with open(path, 'r') as f:
            data = json.load(f)
        docs: Dict[str, Any] = {}
        kg: Dict[str, List[Tuple[int,int,str]]] = {}
        end = len(data) if max_docs is None else min(max_docs, len(data))
        for i, doc in enumerate(data[:end]):
            doc_id = f'docred_{i:04d}'
            docs[doc_id] = {
                'sents': doc.get('sents', []),
                'vertexSet': doc.get('vertexSet', []),
                'labels': doc.get('labels', [])
            }
            rels: List[Tuple[int,int,str]] = []
            for lbl in doc.get('labels', []):
                rels.append((lbl['h'], lbl['t'], lbl['r']))
            kg[doc_id] = rels
        return docs, kg

    def extract_evidence_text(self, doc: Dict[str, Any], h: int, t: int, evidence_ids: List[int] = None) -> str:
        """Build text from evidence sentences; fallback to co-mention sentences if evidence missing.
        Robust to out-of-range entity indices by returning full doc text.
        """
        sents: List[List[str]] = doc.get('sents', [])
        vertex_set: List[List[dict]] = doc.get('vertexSet', [])
        # If entities out of range, fallback to full doc text
        if not (0 <= h < len(vertex_set)) or not (0 <= t < len(vertex_set)):
            return ' '.join(' '.join(s) for s in sents)
        # Collect sentence ids
        sent_ids = set()
        if evidence_ids:
            sent_ids.update(evidence_ids)
        else:
            # Co-mention fallback: any sentence where head and tail mentions appear
            head_sents = {m.get('sent_id', m.get('sent', m.get('sent_id', -1))) for m in vertex_set[h]}
            tail_sents = {m.get('sent_id', m.get('sent', m.get('sent_id', -1))) for m in vertex_set[t]}
            sent_ids = {sid for sid in head_sents.intersection(tail_sents) if 0 <= sid < len(sents)}
            if not sent_ids:
                # fallback: shortest path between first head and tail sentence indices
                if vertex_set[h] and vertex_set[t]:
                    hs = min(max(0, vertex_set[h][0].get('sent_id', 0)), len(sents)-1)
                    ts = min(max(0, vertex_set[t][0].get('sent_id', 0)), len(sents)-1)
                    lo, hi = sorted([hs, ts])
                    sent_ids = set(range(lo, min(hi+1, len(sents))))
        # Build text
        text = ' '.join([' '.join(sents[sid]) for sid in sorted(sent_ids) if 0 <= sid < len(sents)])
        if not text:
            text = ' '.join(' '.join(s) for s in sents)
        return text

    def build_text_with_markers(self, doc: Dict[str, Any], h: int, t: int, evidence_ids: List[int] = None) -> str:
        """Return evidence text with entity markers [E1][/E1], [E2][/E2] where possible."""
        sents: List[List[str]] = doc.get('sents', [])
        vertex_set: List[List[dict]] = doc.get('vertexSet', [])
        if not (0 <= h < len(vertex_set)) or not (0 <= t < len(vertex_set)):
            return ' '.join(' '.join(s) for s in sents)
        # Determine sentence ids to include
        sent_ids = set()
        if evidence_ids:
            sent_ids.update(evidence_ids)
        else:
            head_sents = {m.get('sent_id', m.get('sent', m.get('sent_id', -1))) for m in vertex_set[h]}
            tail_sents = {m.get('sent_id', m.get('sent', m.get('sent_id', -1))) for m in vertex_set[t]}
            inter = head_sents.intersection(tail_sents)
            sent_ids = {sid for sid in inter if 0 <= sid < len(sents)}
            if not sent_ids:
                if vertex_set[h] and vertex_set[t]:
                    hs = min(max(0, vertex_set[h][0].get('sent_id', 0)), len(sents)-1)
                    ts = min(max(0, vertex_set[t][0].get('sent_id', 0)), len(sents)-1)
                    lo, hi = sorted([hs, ts])
                    sent_ids = set(range(lo, min(hi+1, len(sents))))
        if not sent_ids:
            return ' '.join(' '.join(s) for s in sents)
        # Map from sentence to marked tokens
        marked_sents: Dict[int, List[str]] = {}
        for sid in sorted(sent_ids):
            toks = list(sents[sid])
            # mark head mention if in this sentence
            head_mentions = [m for m in vertex_set[h] if m.get('sent_id', m.get('sent', -1)) == sid and 'pos' in m]
            tail_mentions = [m for m in vertex_set[t] if m.get('sent_id', m.get('sent', -1)) == sid and 'pos' in m]
            # apply markers (choose first occurrence per entity)
            # process in reverse index order to keep positions valid after insertions
            inserts = []
            if head_mentions:
                hs, he = head_mentions[0]['pos'][0], head_mentions[0]['pos'][1]
                inserts.append(('E1', hs, he))
            if tail_mentions:
                ts, te = tail_mentions[0]['pos'][0], tail_mentions[0]['pos'][1]
                inserts.append(('E2', ts, te))
            # sort by start desc to avoid index shift
            inserts.sort(key=lambda x: x[1], reverse=True)
            for tag, s0, s1 in inserts:
                s0 = max(0, min(s0, len(toks)))
                s1 = max(s0, min(s1, len(toks)))
                toks[s0:s1] = [f'[{tag}]'] + toks[s0:s1] + [f'[/{tag}]']
            marked_sents[sid] = toks
        return ' '.join(' '.join(marked_sents[sid]) for sid in sorted(marked_sents.keys()))

