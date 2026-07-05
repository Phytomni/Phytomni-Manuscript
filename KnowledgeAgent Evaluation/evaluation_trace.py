#!/usr/bin/env python3
"""Trace QA evaluator.

Input: an Excel/CSV file containing one reference-answer column and one or more
model-answer columns. Output: per-model checkpoints, detailed metrics, summary
tables, and basic plots.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
_RUNTIME_READY = False

METRIC_VERSION = "trace_eval"
ANSWER_CLEAN_VERSION = "structured_answer_clean"

BODY_TAG_NAMES = r"answer|response"
REFERENCE_TAG_NAMES = (
    r"citation\s+sources?|citation\s+index|caption\s+source|"
    r"reference\s+(?:documentation|list|citations|index|sources|attribution|materials)|"
    r"documentation(?:\s+(?:references|trail))?"
)
STRUCTURAL_TAG_NAMES = rf"{BODY_TAG_NAMES}|think|{REFERENCE_TAG_NAMES}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def decode_structural_tokens(value: str) -> str:
    text = html.unescape(str(value or ""))
    pattern = re.compile(
        r"(?is)(?:__LT__|(?<![A-Za-z0-9_])LT__)\s*"
        rf"(?P<closing>/?)\s*(?P<tag>{STRUCTURAL_TAG_NAMES})\s*"
        r"(?:__GT__|GT__(?![A-Za-z0-9_]))"
    )
    text = pattern.sub(lambda m: f"<{m.group('closing')}{m.group('tag')}>", text)
    replacements = {
        r"(?is)__/\s*Answer__": "</Answer>",
        r"(?is)__Answer__": "<Answer>",
        r"(?is)__/\s*Response__": "</Response>",
        r"(?is)__Response__": "<Response>",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text)
    text = re.sub(
        rf"(?is)\*\*\s*(<\s*/?\s*(?:{STRUCTURAL_TAG_NAMES})\b[^>]*>)\s*\*\*",
        r"\1",
        text,
    )
    return text.replace("__PIPE__", " | ").replace("PIPE__", " | ")


def before_reference_section(value: str) -> str:
    return re.split(
        rf"(?is)<\s*/?\s*(?:{REFERENCE_TAG_NAMES})\b[^>]*>",
        value,
        maxsplit=1,
    )[0].strip()


def extract_evaluation_answer(value: Any) -> tuple[str, str]:
    text = decode_structural_tokens("" if value is None else str(value)).strip()
    if not text:
        return "", "empty"

    complete_blocks = [
        m
        for m in re.finditer(
            rf"(?is)<\s*(?P<tag>{BODY_TAG_NAMES})\b[^>]*>"
            rf"(?P<body>.*?)<\s*/\s*(?P=tag)\s*>",
            text,
        )
        if m.group("body").strip()
    ]
    if complete_blocks:
        block = complete_blocks[-1]
        answer = before_reference_section(block.group("body"))
        rule = f"last_complete_{block.group('tag').lower()}_block"
    else:
        openings = list(re.finditer(rf"(?is)<\s*(?P<tag>{BODY_TAG_NAMES})\b[^>]*>", text))
        if openings:
            block = openings[-1]
            answer = before_reference_section(text[block.end() :])
            answer = re.split(r"(?is)<\s*/\s*think\b[^>]*>", answer, maxsplit=1)[0].strip()
            rule = f"unclosed_{block.group('tag').lower()}_block"
        else:
            without_think = re.sub(r"(?is)<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", "", text).strip()
            answer = before_reference_section(without_think)
            rule = "plain_before_reference_or_without_think" if answer != text else "plain_unchanged"

    answer = re.sub(r"(?is)^```(?:[A-Za-z0-9_-]+)?\s*|\s*```$", "", answer).strip()
    answer = re.sub(rf"(?is)<\s*/?\s*(?:{STRUCTURAL_TAG_NAMES})\b[^>]*>", "", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer, rule


def read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    ensure_runtime_dependencies(load_neural=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, engine="openpyxl").fillna("")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).fillna("")
    raise ValueError(f"Unsupported input file extension: {path.suffix}")


def token_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [w for w in cleaned.split() if w.strip()]


def ensure_runtime_dependencies(load_neural: bool = True) -> None:
    global _RUNTIME_READY
    global pd, np, plt, SmoothingFunction, sentence_bleu, Rouge, paired_cosine_distances, tqdm
    if _RUNTIME_READY:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np
        import pandas as _pd
        from nltk.translate.bleu_score import SmoothingFunction as _SmoothingFunction
        from nltk.translate.bleu_score import sentence_bleu as _sentence_bleu
        from rouge import Rouge as _Rouge
        from sklearn.metrics.pairwise import paired_cosine_distances as _paired_cosine_distances
        from tqdm.auto import tqdm as _tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependency. Install pandas, openpyxl, numpy, matplotlib, "
            "nltk, regex, rouge, scikit-learn, and tqdm."
        ) from exc
    pd = _pd
    np = _np
    plt = _plt
    SmoothingFunction = _SmoothingFunction
    sentence_bleu = _sentence_bleu
    Rouge = _Rouge
    paired_cosine_distances = _paired_cosine_distances
    tqdm = _tqdm
    _RUNTIME_READY = True


class TraceEvaluator:
    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        model_columns: list[str],
        reference_column: str = "answer",
        id_column: str = "ID",
        sheet: str | int | None = None,
        bert_model: str = "bert-base-uncased",
        bert_score_num_layers: int | None = 9,
        bleurt_model_path: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        resume: bool = True,
        empty_answer_policy: str = "zero",
        local_files_only: bool = False,
    ) -> None:
        ensure_runtime_dependencies(load_neural=False)
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model_columns = model_columns
        self.reference_column = reference_column
        self.id_column = id_column
        self.bert_model = bert_model
        self.bert_score_num_layers = bert_score_num_layers
        self.bleurt_model_path = bleurt_model_path or ""
        self.batch_size = max(1, int(batch_size))
        self.resume = resume
        self.empty_answer_policy = empty_answer_policy
        self.local_files_only = local_files_only
        self.input_hash = sha256_file(self.input_path)

        try:
            import torch
            from bert_score import BERTScorer
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "Missing neural-metric dependency. Install torch, transformers, bert-score, rouge, "
                "nltk, scikit-learn, pandas, openpyxl, matplotlib, and tqdm before running evaluation."
            ) from exc

        transformers_logging.set_verbosity_error()
        self.torch = torch
        self.BERTScorer = BERTScorer
        self.AutoModel = AutoModel
        self.AutoTokenizer = AutoTokenizer
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)

        self.df = read_table(self.input_path, sheet=sheet)
        missing = {self.id_column, self.reference_column, *self.model_columns}.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        if self.df[self.id_column].astype(str).duplicated().any():
            raise ValueError(f"Duplicate IDs found in column {self.id_column!r}")
        empty_reference = self.df[self.reference_column].astype(str).str.strip().eq("")
        if empty_reference.any():
            ids = self.df.loc[empty_reference, self.id_column].astype(str).tolist()
            raise ValueError(f"Empty reference answers: {ids[:20]}")

        self.clean_answers()
        self.references = self.df[self.reference_column].astype(str).tolist()
        self.ids = self.df[self.id_column].astype(str).tolist()
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.stopwords = {
            "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
            "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "will", "just", "this", "that",
        }

        print(f"Loading encoder: {self.bert_model} on {self.device}", flush=True)
        self.tokenizer = self.AutoTokenizer.from_pretrained(self.bert_model, local_files_only=self.local_files_only)
        self.semantic_model = self.AutoModel.from_pretrained(self.bert_model, local_files_only=self.local_files_only).to(self.device).eval()
        bert_score_kwargs = {
            "model_type": self.bert_model,
            "batch_size": self.batch_size,
            "nthreads": 1,
            "all_layers": False,
            "idf": False,
            "device": self.device,
            "rescale_with_baseline": False,
            "lang": "en",
        }
        if self.bert_score_num_layers is not None:
            bert_score_kwargs["num_layers"] = self.bert_score_num_layers
        self.bert_scorer = self.BERTScorer(
            **bert_score_kwargs,
        )
        self.bleurt_scorer = None
        if self.bleurt_model_path:
            from bleurt import score as bleurt_score

            self.bleurt_scorer = bleurt_score.BleurtScorer(self.bleurt_model_path)
        self.reference_embeddings = self.encode(self.references)
        self.results: dict[str, dict[str, list[Any]]] = {}

    def clean_answers(self) -> None:
        details = []
        summary = []
        coverage = []
        for model in self.model_columns:
            cleaned = []
            counts: Counter[str] = Counter()
            changed = 0
            for _, value in enumerate(self.df[model].astype(str).tolist()):
                answer, rule = extract_evaluation_answer(value)
                cleaned.append(answer)
                counts[rule] += 1
                changed += int(answer != value.strip())
            self.df[model] = cleaned
            empty = sum(not x.strip() for x in cleaned)
            summary.append({"model": model, "changed": changed, "empty_after_clean": empty, **dict(counts)})
            coverage.append(
                {
                    "model": model,
                    "total": len(cleaned),
                    "non_empty": len(cleaned) - empty,
                    "empty": empty,
                    "non_empty_rate": (len(cleaned) - empty) / len(cleaned) if cleaned else 0.0,
                }
            )
            for idx, value in enumerate(cleaned):
                if not value.strip():
                    details.append({"ID": self.df.iloc[idx][self.id_column], "model": model, "rule": "empty_after_clean"})
            if empty and self.empty_answer_policy == "error":
                ids = self.df.loc[self.df[model].astype(str).str.strip().eq(""), self.id_column].astype(str).tolist()
                raise ValueError(f"{model} contains empty answers: {ids[:20]}")
            if empty:
                print(f"{model}: {empty} empty answers will be scored as zero", flush=True)
        self.df.to_excel(self.output_dir / "cleaned_evaluation_input.xlsx", index=False)
        pd.DataFrame(summary).fillna(0).to_csv(self.output_dir / "answer_cleaning_summary.csv", index=False)
        pd.DataFrame(coverage).to_csv(self.output_dir / "answer_coverage.csv", index=False)
        pd.DataFrame(details).to_csv(self.output_dir / "answer_cleaning_details.csv", index=False)

    def signature(self, model: str) -> dict[str, Any]:
        return {
            "metric_version": METRIC_VERSION,
            "answer_clean_version": ANSWER_CLEAN_VERSION,
            "input_sha256": self.input_hash,
            "model": model,
            "row_count": len(self.df),
            "reference_column": self.reference_column,
            "bert_model": self.bert_model,
            "bert_score_num_layers": self.bert_score_num_layers,
            "bleurt_model_path": self.bleurt_model_path,
            "empty_answer_policy": self.empty_answer_policy,
        }

    def checkpoint_path(self, model: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        return self.checkpoint_dir / f"{safe}.json"

    def load_checkpoint(self, model: str) -> dict[int, dict[str, Any]]:
        path = self.checkpoint_path(model)
        if not self.resume or not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != self.signature(model):
            print(f"Ignoring checkpoint with changed signature: {model}", flush=True)
            return {}
        return {int(k): v for k, v in payload.get("records", {}).items()}

    def save_checkpoint(self, model: str, records: dict[int, dict[str, Any]]) -> None:
        atomic_json(
            self.checkpoint_path(model),
            {"signature": self.signature(model), "completed": len(records), "records": {str(k): v for k, v in sorted(records.items())}},
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with self.torch.inference_mode():
                output = self.semantic_model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            chunks.append(pooled.cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 768))

    def bleu(self, reference: str, candidate: str) -> dict[str, float]:
        ref = token_words(reference)
        cand = token_words(candidate)
        if not ref or not cand:
            return {f"bleu-{i}": 0.0 for i in range(1, 5)} | {"bleu-avg": 0.0}
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25,) * 4]
        values = [sentence_bleu([ref], cand, weights=w, smoothing_function=self.smooth) for w in weights]
        return {**{f"bleu-{i + 1}": float(v) for i, v in enumerate(values)}, "bleu-avg": float(np.mean(values))}

    def rouge_scores(self, reference: str, candidate: str) -> dict[str, dict[str, float]]:
        if not reference.strip() or not candidate.strip():
            return {name: {k: 0.0 for k in ("f", "p", "r")} for name in ("rouge-1", "rouge-2", "rouge-l")}
        try:
            return self.rouge.get_scores(candidate, reference)[0]
        except Exception:
            return {name: {k: 0.0 for k in ("f", "p", "r")} for name in ("rouge-1", "rouge-2", "rouge-l")}

    def keyword_coverage(self, reference: str, candidate: str) -> float:
        ref = [w for w in token_words(reference) if len(w) > 1 and w not in self.stopwords]
        cand = [w for w in token_words(candidate) if len(w) > 1 and w not in self.stopwords]
        keywords = {w for w, _ in Counter(ref).most_common(20)}
        return float(len(keywords.intersection(cand)) / len(keywords)) if keywords else 0.0

    def evaluate_batch(self, indices: list[int], model: str) -> dict[int, dict[str, Any]]:
        refs = [self.references[i] for i in indices]
        cands = [str(self.df.iloc[i][model]) for i in indices]
        n = len(indices)
        precision = np.zeros(n)
        recall = np.zeros(n)
        f1 = np.zeros(n)
        semantic = np.zeros(n)
        bleurt = np.zeros(n)
        valid = [i for i, cand in enumerate(cands) if cand.strip()]
        if valid:
            valid_refs = [refs[i] for i in valid]
            valid_cands = [cands[i] for i in valid]
            with self.torch.inference_mode():
                p, r, f = self.bert_scorer.score(valid_cands, valid_refs, batch_size=self.batch_size)
            valid_embeddings = self.encode(valid_cands)
            valid_semantic = 1.0 - paired_cosine_distances(self.reference_embeddings[[indices[i] for i in valid]], valid_embeddings)
            valid_bleurt = self.bleurt_scorer.score(references=valid_refs, candidates=valid_cands) if self.bleurt_scorer else [0.0] * len(valid)
            for offset, valid_offset in enumerate(valid):
                precision[valid_offset] = float(p[offset])
                recall[valid_offset] = float(r[offset])
                f1[valid_offset] = float(f[offset])
                semantic[valid_offset] = float(valid_semantic[offset])
                bleurt[valid_offset] = float(valid_bleurt[offset])

        records = {}
        for offset, index in enumerate(indices):
            records[index] = {
                "id": str(self.df.iloc[index][self.id_column]),
                "missing_answer": not cands[offset].strip(),
                "bleu": self.bleu(refs[offset], cands[offset]),
                "rouge": self.rouge_scores(refs[offset], cands[offset]),
                "keyword_coverage": self.keyword_coverage(refs[offset], cands[offset]),
                "bert_score": {"precision": float(precision[offset]), "recall": float(recall[offset]), "f1": float(f1[offset])},
                "semantic_similarity": float(semantic[offset]),
                "bleurt": float(bleurt[offset]),
            }
        return records

    def evaluate(self) -> dict[str, dict[str, list[Any]]]:
        total = len(self.df) * len(self.model_columns)
        with tqdm(total=total, desc="TOTAL", dynamic_ncols=True) as overall:
            for model in self.model_columns:
                records = self.load_checkpoint(model)
                overall.update(len(records))
                pending = [i for i in range(len(self.df)) if i not in records]
                for start in tqdm(range(0, len(pending), self.batch_size), desc=model, leave=False, dynamic_ncols=True):
                    indices = pending[start : start + self.batch_size]
                    records.update(self.evaluate_batch(indices, model))
                    self.save_checkpoint(model, records)
                    overall.update(len(indices))
                self.save_checkpoint(model, records)

        self.results = {}
        for model in self.model_columns:
            records = self.load_checkpoint(model)
            ordered = [records[i] for i in range(len(self.df))]
            self.results[model] = {
                "ids": [r["id"] for r in ordered],
                "bleu": [r["bleu"] for r in ordered],
                "rouge": [r["rouge"] for r in ordered],
                "keyword_coverage": [r["keyword_coverage"] for r in ordered],
                "bert_score": [r["bert_score"] for r in ordered],
                "semantic_similarity": [r["semantic_similarity"] for r in ordered],
                "bleurt": [r["bleurt"] for r in ordered],
            }
        atomic_json(self.output_dir / "detailed_results.json", self.results)
        return self.results

    def aggregate(self) -> pd.DataFrame:
        if not self.results:
            self.evaluate()
        rows = {}
        for model, result in self.results.items():
            rows[model] = {
                "bleu-1": float(np.mean([x["bleu-1"] for x in result["bleu"]])),
                "bleu-2": float(np.mean([x["bleu-2"] for x in result["bleu"]])),
                "bleu-3": float(np.mean([x["bleu-3"] for x in result["bleu"]])),
                "bleu-4": float(np.mean([x["bleu-4"] for x in result["bleu"]])),
                "bleu-avg": float(np.mean([x["bleu-avg"] for x in result["bleu"]])),
                "rouge-1-f": float(np.mean([x["rouge-1"]["f"] for x in result["rouge"]])),
                "rouge-2-f": float(np.mean([x["rouge-2"]["f"] for x in result["rouge"]])),
                "rouge-l-f": float(np.mean([x["rouge-l"]["f"] for x in result["rouge"]])),
                "keyword_coverage": float(np.mean(result["keyword_coverage"])),
                "bert_score-precision": float(np.mean([x["precision"] for x in result["bert_score"]])),
                "bert_score-recall": float(np.mean([x["recall"] for x in result["bert_score"]])),
                "bert_score-f1": float(np.mean([x["f1"] for x in result["bert_score"]])),
                "semantic_similarity": float(np.mean(result["semantic_similarity"])),
                "bleurt": float(np.mean(result["bleurt"])),
            }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.to_csv(self.output_dir / "aggregated_results.csv")
        return df

    def rankings(self, aggregated: pd.DataFrame) -> pd.DataFrame:
        ranks = aggregated.rank(ascending=False)
        ranks["average_rank"] = ranks.mean(axis=1)
        ranks = ranks.sort_values("average_rank")
        ranks.to_csv(self.output_dir / "model_rankings.csv")
        return ranks

    def export_per_question_metrics(self) -> pd.DataFrame:
        if not self.results:
            self.evaluate()
        out = self.df[[self.id_column, self.reference_column]].copy()
        for model, result in self.results.items():
            out[f"{model}__Answer"] = self.df[model].tolist()
            out[f"{model}__BERTScore F1"] = [x["f1"] for x in result["bert_score"]]
            out[f"{model}__BERTScore precision"] = [x["precision"] for x in result["bert_score"]]
            out[f"{model}__BERTScore recall"] = [x["recall"] for x in result["bert_score"]]
            for key in ["bleu-1", "bleu-2", "bleu-3", "bleu-4"]:
                out[f"{model}__{key.upper()}"] = [x[key] for x in result["bleu"]]
            out[f"{model}__ROUGE-1 F1"] = [x["rouge-1"]["f"] for x in result["rouge"]]
            out[f"{model}__ROUGE-2 F1"] = [x["rouge-2"]["f"] for x in result["rouge"]]
            out[f"{model}__ROUGE-L F1"] = [x["rouge-l"]["f"] for x in result["rouge"]]
        out.to_excel(self.output_dir / "per_question_metrics.xlsx", index=False)
        out.to_csv(self.output_dir / "per_question_metrics.csv", index=False)
        return out

    def plot(self, aggregated: pd.DataFrame, rankings: pd.DataFrame) -> None:
        metric_files = {
            "bleu-avg": "bleu_avg_comparison.png",
            "rouge-l-f": "rouge_l_f_comparison.png",
            "bert_score-f1": "bert_score_f1_comparison.png",
            "semantic_similarity": "semantic_similarity_comparison.png",
            "bleurt": "bleurt_comparison.png",
        }
        for metric, filename in metric_files.items():
            plt.figure(figsize=(max(8, len(aggregated) * 0.55), 5))
            aggregated[metric].sort_values(ascending=False).plot(kind="bar")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(self.output_dir / filename, dpi=300)
            plt.close()

        plt.figure(figsize=(max(8, len(rankings) * 0.55), 5))
        rankings["average_rank"].sort_values().plot(kind="bar")
        plt.ylabel("average rank (lower is better)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "average_rankings.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, max(6, len(aggregated) * 0.35)))
        plt.imshow(aggregated.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(label="score")
        plt.yticks(range(len(aggregated.index)), aggregated.index)
        plt.xticks(range(len(aggregated.columns)), aggregated.columns, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "scores_heatmap.png", dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trace-style QA answers with BLEU, ROUGE, BERTScore, semantic similarity, and optional BLEURT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .xlsx/.xls/.csv file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for evaluation outputs.")
    parser.add_argument("--models", required=True, nargs="+", help="Model answer columns to evaluate.")
    parser.add_argument("--reference-column", default="answer", help="Reference answer column.")
    parser.add_argument("--id-column", default="ID", help="Question ID column.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index. Defaults to first sheet.")
    parser.add_argument("--bert-model", default="bert-base-uncased", help="HuggingFace model name or local path for BERTScore/semantic similarity.")
    parser.add_argument("--bleurt-model-path", default=None, help="Optional local BLEURT checkpoint path. If omitted, BLEURT is set to 0.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device.")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size for neural metrics.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and recompute.")
    parser.add_argument("--empty-answer-policy", default="zero", choices=["zero", "error"], help="How to handle empty model answers after cleaning.")
    parser.add_argument("--local-files-only", action="store_true", help="Load HuggingFace models from local cache/path only.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    evaluator = TraceEvaluator(
        input_path=args.input,
        output_dir=args.output_dir,
        model_columns=args.models,
        reference_column=args.reference_column,
        id_column=args.id_column,
        sheet=sheet,
        bert_model=args.bert_model,
        bleurt_model_path=args.bleurt_model_path,
        device=args.device,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        empty_answer_policy=args.empty_answer_policy,
        local_files_only=args.local_files_only,
    )
    evaluator.evaluate()
    aggregated = evaluator.aggregate()
    rankings = evaluator.rankings(aggregated)
    evaluator.export_per_question_metrics()
    if not args.no_plots:
        evaluator.plot(aggregated, rankings)
    print(f"Done. Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
