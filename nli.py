"""NLI 모델 래퍼 — 문장 수준 신규 정보 탐지"""

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


class NLIModel:
    def __init__(self, model_name="nomad0884/korean-nli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_name = model_name
        print(f"[NLI] 모델 로드: {model_name} ({self.device})")

    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits, dim=1)[0]

        label = LABEL_NAMES[probs.argmax().item()]
        scores = {n: round(p.item(), 4) for n, p in zip(LABEL_NAMES, probs)}
        return {"label": label, "scores": scores}

    @staticmethod
    def split_sentences(text):
        text = text.replace('\n\n', '. ').replace('\n', '. ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def detect_novel(self, existing_text, new_text, threshold=0.5):
        """
        새 기사에서 기존 기사에 없는 신규 정보 문장 탐지

        Args:
            existing_text: 기존 기사 본문
            new_text: 새 기사 본문
            threshold: entailment 확률 임계값

        Returns:
            {"novel": [...], "redundant": [...], "novelty_ratio": float}
        """
        p_sents = self.split_sentences(existing_text)
        h_sents = self.split_sentences(new_text)

        if not p_sents or not h_sents:
            return {"novel": [], "redundant": [], "novelty_ratio": 0, "total": 0}

        novel, redundant = [], []

        for h in h_sents:
            max_ent = 0.0
            best_match = ""

            for p in p_sents:
                result = self.predict(premise=p, hypothesis=h)
                ent = result["scores"]["entailment"]
                if ent > max_ent:
                    max_ent = ent
                    best_match = p

            info = {
                "sentence": h,
                "max_entailment": round(max_ent, 4),
                "best_match": best_match
            }

            if max_ent < threshold:
                novel.append(info)
            else:
                redundant.append(info)

        total = len(h_sents)
        return {
            "novel": novel,
            "redundant": redundant,
            "novel_count": len(novel),
            "redundant_count": len(redundant),
            "total": total,
            "novelty_ratio": round(len(novel) / total, 4) if total > 0 else 0
        }
