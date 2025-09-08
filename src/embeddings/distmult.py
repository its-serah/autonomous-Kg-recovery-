from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

class DistMultScorer(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eh = self.E(h)
        er = self.R(r)
        et = self.E(t)
        # DistMult: <eh, er, et>
        return torch.sum(eh * er * et, dim=-1)

class KGETrainer:
    def __init__(self, dim: int = 64, lr: float = 1e-2, neg_ratio: int = 2, epochs: int = 2, device: str = None):
        self.dim = dim
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.ent2id: Dict[str,int] = {}
        self.rel2id: Dict[str,int] = {}

    def _get_eid(self, ent: str) -> int:
        if ent not in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def _get_rid(self, rel: str) -> int:
        if rel not in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def fit(self, triples: List[Tuple[str,str,str]]):
        # Build ids
        for h,r,t in triples:
            self._get_eid(h); self._get_eid(t); self._get_rid(r)
        self.model = DistMultScorer(len(self.ent2id), len(self.rel2id), dim=self.dim).to(self.device)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        # Prepare data
        triples_id = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h,r,t in triples]
        all_ents = list(range(len(self.ent2id)))
        for epoch in range(self.epochs):
            random.shuffle(triples_id)
            total_loss = 0.0
            for (h,r,t) in triples_id:
                # build batch with negatives
                h_tensor = []
                r_tensor = []
                t_tensor = []
                y = []
                # positive
                h_tensor.append(h); r_tensor.append(r); t_tensor.append(t); y.append(1.0)
                # negatives
                for _ in range(self.neg_ratio):
                    if random.random() < 0.5:
                        hneg = random.choice(all_ents)
                        h_tensor.append(hneg); r_tensor.append(r); t_tensor.append(t); y.append(0.0)
                    else:
                        tneg = random.choice(all_ents)
                        h_tensor.append(h); r_tensor.append(r); t_tensor.append(tneg); y.append(0.0)
                h_tensor = torch.tensor(h_tensor, dtype=torch.long, device=self.device)
                r_tensor = torch.tensor(r_tensor, dtype=torch.long, device=self.device)
                t_tensor = torch.tensor(t_tensor, dtype=torch.long, device=self.device)
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                opt.zero_grad()
                logits = self.model.score_triples(h_tensor, r_tensor, t_tensor)
                loss = bce(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            # print(f"KGE epoch {epoch+1}, loss={total_loss/len(triples_id):.4f}")

    def score(self, h: str, r: str, t: str) -> float:
        if (h not in self.ent2id) or (t not in self.ent2id) or (r not in self.rel2id):
            return 0.0
        self.model.eval()
        with torch.no_grad():
            h_id = torch.tensor([self.ent2id[h]], dtype=torch.long, device=self.device)
            r_id = torch.tensor([self.rel2id[r]], dtype=torch.long, device=self.device)
            t_id = torch.tensor([self.ent2id[t]], dtype=torch.long, device=self.device)
            logit = self.model.score_triples(h_id, r_id, t_id).item()
            # map logit to [0,1]
            return 1/(1+math.exp(-logit))

