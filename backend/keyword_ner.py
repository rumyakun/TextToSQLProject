import json
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .keyword_config import MODEL_DIR


_predictor_lock = threading.Lock()
_predictor: "CourseNERPredictor | None" = None


class CourseNERPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR, max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

        with open(self.model_dir / "labels.json", "r", encoding="utf-8") as file:
            label_data = json.load(file)

        self.id_to_label = {
            int(key): value
            for key, value in label_data["id_to_label"].items()
        }

    @torch.no_grad()
    def extract(self, text: str) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_scores = torch.max(probs, dim=-1).values.tolist()

        token_predictions = []
        for pred_id, score, (start, end) in zip(pred_ids, pred_scores, offsets):
            if start == end:
                continue
            token_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "label": self.id_to_label[pred_id],
                    "score": float(score),
                }
            )

        return self._merge_bio(text, token_predictions)

    def _merge_bio(self, text: str, token_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities = []
        current = None

        for token in token_predictions:
            label = token["label"]
            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or current is None or current["label"] != entity_type:
                if current:
                    entities.append(current)
                current = {
                    "label": entity_type,
                    "start": token["start"],
                    "end": token["end"],
                    "scores": [token["score"]],
                }
                continue

            current["end"] = token["end"]
            current["scores"].append(token["score"])

        if current:
            entities.append(current)

        result = []
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            avg_score = sum(entity["scores"]) / len(entity["scores"])
            result.append(
                {
                    "text": text[start:end],
                    "label": entity["label"],
                    "start": start,
                    "end": end,
                    "score": round(avg_score, 4),
                }
            )

        return result


def get_predictor(model_dir: str | Path = MODEL_DIR, max_length: int = 128) -> CourseNERPredictor:
    global _predictor

    with _predictor_lock:
        if _predictor is None:
            _predictor = CourseNERPredictor(model_dir=model_dir, max_length=max_length)
        return _predictor
