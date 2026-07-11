#!/usr/bin/env python3
"""ID benchmark evaluator.

Evaluate gene-name answers against pipe-separated gold aliases. The matching
logic follows the original species-aware ID benchmark rule: a prediction is
correct when it contains any gold alias, with simple species prefix variants
for Arabidopsis, rice, maize, and wheat.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MAIN_SPECIES = [
    "Oryza sativa",
    "Arabidopsis thaliana",
    "Zea mays",
    "Glycine max",
    "Triticum Aestivum",
]
CORE_COLUMNS = {
    "ID",
    "Species",
    "species",
    "Query",
    "query",
    "Answer",
    "answer",
    "Reference",
    "reference",
    "Prompts",
    "Prompts_modified",
    "prompt",
}


def ensure_dependencies() -> None:
    global pd, plt
    try:
        import pandas as _pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError as exc:
        raise RuntimeError("Install pandas, openpyxl, and matplotlib before running this evaluator.") from exc
    pd = _pd
    plt = _plt


def decode_tokens(value: Any) -> str:
    text = html.unescape("" if value is None else str(value))
    replacements = {
        "__LT__": "<",
        "__GT__": ">",
        "__PIPE__": " | ",
        "PIPE__": " | ",
        "__SEMI__": ";",
        "SEMI__": ";",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def strip_think(text: str) -> str:
    text = re.sub(r"(?is)<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", "", text)
    text = re.sub(r"(?is)<\s*think\b[^>]*>.*$", "", text)
    return text.strip()


def strip_wrappers(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?is)^```(?:json|text|python)?\s*", "", text)
    text = re.sub(r"(?is)\s*```$", "", text).strip()
    text = re.sub(r"(?is)^Output\s*:\s*", "", text).strip()
    text = re.sub(r"(?is)^Final\s*Answer\s*[:：]\s*", "", text).strip()
    text = re.sub(r"(?is)^Answer\s*[:：]\s*", "", text).strip()
    return text


def parse_gene_json(text: str) -> str:
    candidates = [text]
    candidates.extend(m.group(0) for m in re.finditer(r"\{.*?\}", text, flags=re.S))
    keys = ["Gene Names", "Gene names", "gene_names", "Gene_Names", "genes", "Genes", "answer", "Answer"]
    for candidate in candidates:
        for raw in (candidate, candidate.replace("'", '"')):
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                for key in keys:
                    if key in obj and str(obj[key]).strip():
                        return str(obj[key]).strip()
    return ""


def extract_tag_block(text: str) -> str:
    blocks = list(
        re.finditer(
            r"(?is)<\s*(?:answer|response)\b[^>]*>(.*?)<\s*/\s*(?:answer|response)\s*>",
            text,
        )
    )
    if blocks:
        return blocks[-1].group(1).strip()
    openings = list(re.finditer(r"(?is)<\s*(?:answer|response)\b[^>]*>", text))
    if openings:
        return text[openings[-1].end() :].strip()
    return text


def extract_label(text: str) -> str:
    matches = list(re.finditer(r"(?is)(?:Gene\s*Names?|Genes?|基因名)\s*[:：]\s*", text))
    if not matches:
        return ""
    tail = text[matches[-1].end() :].strip()
    tail = re.split(r"(?is)(?:Reference|Reason|Explanation|Note|分析|理由)\s*[:：]", tail, maxsplit=1)[0]
    return tail.strip().strip('"\'{}[]`')


def last_gene_like_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    gene_pat = re.compile(r"\b(?:AT\dG\d+|Os\w+|Zm\w+|Glyma\.|TraesCS|Ta\w+|PUB\d+|NAC\d+|WRKY\d+)\b", re.I)
    for line in reversed(lines[-5:]):
        if "|" in line or gene_pat.search(line):
            return line.strip('"\'{}[]`')
    return text.strip().strip('"\'{}[]`')


def clean_gene_answer(value: Any) -> str:
    text = decode_tokens(value)
    text = strip_think(text)
    if not text.strip():
        return ""
    text = extract_tag_block(text)
    text = re.split(
        r"(?is)<\s*(?:citation\s+sources?|reference\s+(?:documentation|list|citations|sources)|citation\s+index)\b[^>]*>",
        text,
        maxsplit=1,
    )[0]
    text = strip_wrappers(text)
    parsed = parse_gene_json(text)
    if parsed:
        text = parsed
    else:
        labelled = extract_label(text)
        text = labelled if labelled else last_gene_like_line(text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = re.sub(r"(?is)\s+(?:Reference|Reason|Explanation|Note|分析|理由)\s*[:：].*$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip().strip('"\'{}[]`')
    return text


def read_table(path: Path, sheet: str | int | None = None):
    ensure_dependencies()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, engine="openpyxl").fillna("")
    if suffix == ".csv":
        return pd.read_csv(path).fillna("")
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def split_gold(answer: Any) -> list[str]:
    parts = re.split(r"\|", "" if answer is None else str(answer))
    return [p.strip() for p in parts if p and p.strip()]


def normalize_species(value: Any) -> str:
    text = str(value).strip()
    lookup = {x.lower(): x for x in MAIN_SPECIES}
    return lookup.get(text.lower(), text)


def species_group(value: Any, main_species: list[str]) -> str:
    species = normalize_species(value)
    lookup = {x.lower(): x for x in main_species}
    return lookup.get(species.lower(), "Mix")


def species_prefixes(species: Any) -> list[str]:
    sp = normalize_species(species)
    if sp == "Arabidopsis thaliana":
        return ["AT"]
    if sp == "Oryza sativa":
        return ["Os"]
    if sp == "Triticum Aestivum":
        return ["Ta"]
    if sp == "Zea mays":
        return ["Zm"]
    return []


def candidate_aliases(gene: str, species: Any) -> list[str]:
    gene = gene.strip()
    if not gene:
        return []
    out = {gene}
    for prefix in species_prefixes(species):
        out.add(re.sub(rf"^{prefix}", "", gene, flags=re.I))
        out.add(gene if re.match(rf"^{prefix}", gene, flags=re.I) else prefix + gene)
    return [x for x in out if x]


def pattern_match(pattern: str, text: str, mode: str) -> bool:
    if not pattern or not str(text).strip():
        return False
    if mode == "literal":
        return pattern.lower() in str(text).lower()
    try:
        return bool(re.search(pattern, str(text), flags=re.I))
    except re.error:
        return bool(re.search(re.escape(pattern), str(text), flags=re.I))


def check_match(gold_answer: Any, prediction: Any, species: Any, mode: str = "regex") -> bool:
    pred = "" if prediction is None else str(prediction)
    if not pred.strip():
        return False
    for gene in split_gold(gold_answer):
        for candidate in candidate_aliases(gene, species):
            if pattern_match(candidate, pred, mode=mode):
                return True
    return False


def infer_model_columns(columns: list[str]) -> list[str]:
    answer_cols = [c for c in columns if c.endswith("__answer")]
    if answer_cols:
        return answer_cols
    return [
        c
        for c in columns
        if c not in CORE_COLUMNS
        and not c.endswith("__correct")
        and not c.endswith("_check")
        and not c.startswith("Unnamed:")
    ]


def summarize_accuracy(df, model_columns: list[str], species_col: str, main_species: list[str]):
    rows = []
    species_rows = []
    for model in model_columns:
        check_col = f"{model}__correct"
        total = len(df)
        correct = int(df[check_col].sum())
        rows.append({"model": model, "correct": correct, "total": total, "accuracy": correct / total if total else 0.0})
        for species, sub in df.groupby("species_group", dropna=False):
            t = len(sub)
            c = int(sub[check_col].sum())
            species_rows.append({"model": model, "species": species, "correct": c, "total": t, "accuracy": c / t if t else 0.0})
    by_model = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    by_species_model = pd.DataFrame(species_rows)
    order = main_species + ["Mix"]
    by_species_model["species"] = pd.Categorical(by_species_model["species"], categories=order, ordered=True)
    by_species_model = by_species_model.sort_values(["species", "model"]).reset_index(drop=True)
    by_species_model["species"] = by_species_model["species"].astype(str)
    by_species_rows = []
    for species, sub in df.groupby("species_group", dropna=False):
        total = len(sub) * len(model_columns)
        correct = int(sum(sub[f"{m}__correct"].sum() for m in model_columns))
        by_species_rows.append({"species": species, "correct": correct, "total": total, "accuracy": correct / total if total else 0.0})
    by_species = pd.DataFrame(by_species_rows)
    by_species["species"] = pd.Categorical(by_species["species"], categories=order, ordered=True)
    by_species = by_species.sort_values("species").reset_index(drop=True)
    by_species["species"] = by_species["species"].astype(str)
    overall_correct = int(sum(df[f"{m}__correct"].sum() for m in model_columns))
    overall_total = len(df) * len(model_columns)
    overall = pd.DataFrame(
        [{"correct": overall_correct, "total": overall_total, "accuracy": overall_correct / overall_total if overall_total else 0.0}]
    )
    return overall, by_model, by_species, by_species_model


def plot_accuracy(by_model, by_species_model, output_dir: Path) -> None:
    if by_model.empty:
        return
    plt.figure(figsize=(max(8, len(by_model) * 0.55), 5))
    by_model.set_index("model")["accuracy"].sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_model.png", dpi=300)
    plt.close()

    if not by_species_model.empty:
        pivot = by_species_model.pivot(index="species", columns="model", values="accuracy")
        plt.figure(figsize=(max(10, len(pivot.columns) * 0.55), max(5, len(pivot.index) * 0.45)))
        plt.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(label="accuracy")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_by_species_model_heatmap.png", dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ID benchmark gene-name answers with species-aware alias matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .xlsx/.xls/.csv file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output tables and plots.")
    parser.add_argument("--models", nargs="+", default=None, help="Model answer columns. Defaults to columns ending with __answer.")
    parser.add_argument("--answer-column", default="Answer", help="Gold pipe-separated answer column.")
    parser.add_argument("--species-column", default="Species", help="Species column.")
    parser.add_argument("--id-column", default="ID", help="Question ID column.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index. Defaults to first sheet.")
    parser.add_argument("--main-species", default=",".join(MAIN_SPECIES), help="Comma-separated species kept separate; all others become Mix.")
    parser.add_argument("--match-mode", choices=["regex", "literal"], default="regex", help="Gold alias matching mode.")
    parser.add_argument("--empty-answer-fill", default="", help="Optional text used only in output for empty cleaned answers.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    df = read_table(args.input, sheet=sheet)
    missing_base = {args.id_column, args.species_column, args.answer_column}.difference(df.columns)
    if missing_base:
        raise ValueError(f"Missing required columns: {sorted(missing_base)}")
    model_columns = args.models if args.models else infer_model_columns(df.columns.tolist())
    missing_models = [c for c in model_columns if c not in df.columns]
    if missing_models:
        raise ValueError(f"Missing model columns: {missing_models}")
    if not model_columns:
        raise ValueError("No model columns found. Pass --models explicitly.")

    main_species = [x.strip() for x in args.main_species.split(",") if x.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detail = df.copy()
    detail["species_group"] = [species_group(x, main_species) for x in detail[args.species_column]]
    cleaning_rows = []
    for model in model_columns:
        cleaned_col = f"{model}__clean"
        check_col = f"{model}__correct"
        cleaned = [clean_gene_answer(x) for x in detail[model]]
        if args.empty_answer_fill:
            cleaned_out = [x if x.strip() else args.empty_answer_fill for x in cleaned]
        else:
            cleaned_out = cleaned
        detail[cleaned_col] = cleaned_out
        detail[check_col] = [
            check_match(gold, pred, species, mode=args.match_mode)
            for gold, pred, species in zip(detail[args.answer_column], cleaned, detail[args.species_column])
        ]
        cleaning_rows.append(
            {
                "model": model,
                "total": len(cleaned),
                "empty_after_clean": sum(not x.strip() for x in cleaned),
                "changed_by_cleaning": sum(str(a).strip() != b.strip() for a, b in zip(detail[model], cleaned)),
            }
        )

    overall, by_model, by_species, by_species_model = summarize_accuracy(
        detail, model_columns=model_columns, species_col=args.species_column, main_species=main_species
    )
    detail.to_excel(output_dir / "id_evaluation_detail.xlsx", index=False)
    detail.to_csv(output_dir / "id_evaluation_detail.csv", index=False)
    pd.DataFrame(cleaning_rows).to_csv(output_dir / "answer_cleaning_summary.csv", index=False)
    overall.to_excel(output_dir / "accuracy_overall.xlsx", index=False)
    by_model.to_excel(output_dir / "accuracy_by_model.xlsx", index=False)
    by_species.to_excel(output_dir / "accuracy_by_species.xlsx", index=False)
    by_species_model.to_excel(output_dir / "accuracy_by_species_model.xlsx", index=False)
    with pd.ExcelWriter(output_dir / "id_accuracy_summary.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_model.to_excel(writer, sheet_name="by_model", index=False)
        by_species.to_excel(writer, sheet_name="by_species", index=False)
        by_species_model.to_excel(writer, sheet_name="by_species_model", index=False)
        pd.DataFrame(cleaning_rows).to_excel(writer, sheet_name="cleaning", index=False)
    if not args.no_plots:
        plot_accuracy(by_model, by_species_model, output_dir)
    print(f"Done. Evaluated {len(model_columns)} model columns and {len(detail)} rows.")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
