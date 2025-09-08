from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

@dataclass
class TripleExampleHF:
    text: str
    relation: str
    label: int

class TripleDataset(Dataset):
    def __init__(self, tokenizer, examples: List[TripleExampleHF], max_length: int = 384, rel_token: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rel_token = rel_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"REL_{ex.relation} \n {ex.text}" if self.rel_token else ex.text
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length')
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if ex.label is not None:
            item['labels'] = torch.tensor(ex.label, dtype=torch.long)
        return item

class HFDocREVerifier:
    def __init__(self, model_name: str = 'distilbert-base-uncased', max_length: int = 384, epochs: int = 1, batch_size: int = 16):
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        self.fitted = False

    def fit(self, examples: List[TripleExampleHF]):
        # subsample if too large for speed
        if len(examples) > 40000:
            examples = examples[:40000]
        ds = TripleDataset(self.tokenizer, examples, max_length=self.max_length)
        args = TrainingArguments(
            output_dir='./.hf_out',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_steps=100,
            learning_rate=5e-5,
            save_strategy='no',
            report_to=[],
            label_smoothing_factor=0.1
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=ds)
        trainer.train()
        self.fitted = True

    def predict_proba(self, texts_relations: List[Tuple[str,str]]) -> np.ndarray:
        assert self.fitted
        inputs = [f"REL_{r} \n {t}" for (t, r) in texts_relations]
        enc = self.tokenizer(inputs, truncation=True, max_length=self.max_length, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
        return probs

    def calibrate_thresholds(self, dev_examples: List[TripleExampleHF]) -> Dict[str, float]:
        # Compute per-relation thresholds
        rel2pairs: Dict[str, List[Tuple[float,int]]] = {}
        for ex in dev_examples:
            p = self.predict_proba([(ex.text, ex.relation)])[0]
            rel2pairs.setdefault(ex.relation, []).append((p, ex.label))
        rel2thr: Dict[str, float] = {}
        for r, pairs in rel2pairs.items():
            if not pairs:
                rel2thr[r] = 0.5
                continue
            scores = np.array([s for s,_ in pairs])
            labels = np.array([y for _,y in pairs])
            best_f1, best_t = -1.0, 0.5
            for t in np.linspace(0.1, 0.9, 17):
                preds = (scores >= t).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))
                f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            rel2thr[r] = float(best_t)
        return rel2thr

