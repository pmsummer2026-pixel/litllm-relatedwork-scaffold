"""
Convert the openscholar-mini 10-paper corpus (data/parsed/*.json) into the
abstract / related_work / ref_abstract schema expected by litllm-mini's
generation pipeline (see generation/autoreview/models/plan_based_generation.py).

Each parsed paper JSON can be in one of two shapes:
  1. Flat (already normalized): {"paper_id", "title", "authors", "year",
     "abstract", "sections": [...]}
  2. Raw ingest output (as produced by openscholar-mini's PDF parsing step):
     {"filename", "metadata": {"arxiv", "title", "authors", "year",
     "abstract", ...}, "fullText": {...}, "citations": [...]}

This script normalizes either shape, builds one ref_abstract entry per paper
(cite_1..cite_N mapped to each paper's own abstract), and writes a single-row
HuggingFace dataset to --out_dir that can be loaded with
datasets.load_from_disk() and passed straight to litllm-mini's generation
script (see get_dataset() in plan_based_generation.py, which now accepts a
local directory path).

No query/target abstract is required. The row's "abstract" field is left
blank by default, so the model is asked to write a related-work section
synthesizing the 10 papers' own abstracts rather than relating them to one
external target paper. Pass --query_abstract / --query_abstract_file only if
you want to anchor the section to a specific paper's abstract instead.

Usage:
    python openscholar_mini/to_litllm_schema.py \\
        --papers_dir openscholar_mini/data/parsed \\
        --out_dir openscholar_mini/litllm_dataset
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def _extract_full_text(raw):
    ft = raw.get("fullText")
    if isinstance(ft, str):
        return ft
    if isinstance(ft, dict):
        return ft.get("fullText", "") or ""
    return ""


def normalize_paper(raw, fallback_id):
    """Return a flat dict with paper_id/title/authors/year/abstract,
    regardless of whether raw is already flat or is raw ingest output."""
    if "abstract" in raw and "paper_id" in raw:
        return raw
    meta = raw.get("metadata", {})
    return {
        "paper_id": meta.get("arxiv") or fallback_id,
        "title": meta.get("title") or fallback_id,
        "authors": meta.get("authors") or [],
        "year": meta.get("year"),
        "abstract": meta.get("abstract") or _extract_full_text(raw)[:1000],
    }


def load_papers(papers_dir):
    papers = []
    for path in sorted(Path(papers_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        fallback_id = path.stem.replace("_metadata", "")
        papers.append(normalize_paper(raw, fallback_id))
    return papers


def build_ref_abstract(papers):
    cite_tags = [f"@cite_{i + 1}" for i in range(len(papers))]
    abstracts = [p["abstract"] for p in papers]
    return {"cite_N": cite_tags, "abstract": abstracts}


def read_text_arg(inline_value, file_value):
    if file_value:
        return Path(file_value).read_text(encoding="utf-8").strip()
    return (inline_value or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_dir", default="openscholar_mini/data/parsed")
    parser.add_argument("--out_dir", default="openscholar_mini/litllm_dataset")
    parser.add_argument("--query_abstract", default="")
    parser.add_argument("--query_abstract_file", default="")
    parser.add_argument("--related_work", default="")
    parser.add_argument("--related_work_file", default="")
    args = parser.parse_args()

    papers = load_papers(args.papers_dir)
    if not papers:
        raise SystemExit(f"No paper JSON files found in {args.papers_dir}")
    print(f"Loaded {len(papers)} papers from {args.papers_dir}")
    for p in papers:
        print(f"  - {p['paper_id']}: {p['title'][:80]}")
        if not p["abstract"]:
            print(f"    WARNING: empty abstract for {p['paper_id']}")

    # No query abstract is required: left blank by default so the model
    # writes a related-work section covering the 10 papers themselves.
    query_abstract = read_text_arg(args.query_abstract, args.query_abstract_file)
    related_work = read_text_arg(args.related_work, args.related_work_file)

    ref_abstract = build_ref_abstract(papers)

    row = {
        "abstract": query_abstract,
        "related_work": related_work,
        "ref_abstract": ref_abstract,
    }

    dataset = Dataset.from_list([row])
    dataset.save_to_disk(args.out_dir)
    print(f"Saved 1-row dataset with {len(papers)} candidate references to {args.out_dir}")
    print("Load it in litllm-mini via: datasets.load_from_disk(args.out_dir)")


if __name__ == "__main__":
    main()
