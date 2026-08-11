# litllm-relatedwork-scaffold

Minimal, runnable scaffold extracted from [litllm-mini](https://github.com/pmsummer2026-pixel/litllm-mini) for generating a related-works section from the 10-paper `openscholar_mini` corpus, using Claude instead of OpenAI/Llama.

## What's here

- `generation/autoreview/` — only the modules `plan_based_generation.py` actually imports (data_utils, ml_utils, pipeline, chatgpt_model, base_model, llama2_zero_shot, parse_all_args, langchain_openai_agent, anthropic_agent, anyscale_endpoint, evaluation/compute_score, resources/prompts.json).
- `generation/autoreview/models/anthropic_agent.py` — a Claude API wrapper matching the interface plan_based_generation.py expects (added; not part of upstream litllm).
- `generation/autoreview/models/plan_based_generation.py` — patched so `--model_name` accepts Claude models and `--dataset_name` accepts a local dataset directory (via `load_from_disk`), on top of everything upstream already supports.
- `generation/autoreview/models/toolkit_utils.py` — patched so it only requires `HF_TOKEN` when actually logging into Hugging Face (`do_hf_login=True`), instead of unconditionally.
- `openscholar_mini/data/parsed/*.json` — the 10 papers' parsed metadata/full text, copied from openscholar-mini.
- `openscholar_mini/to_litllm_schema.py` — converts those 10 papers into the `abstract`/`related_work`/`ref_abstract` schema litllm expects, with `abstract` left blank (no single target paper) and `ref_abstract` populated with all 10 real abstracts.

## Setup

```bash
pip install -r generation/requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Run

```bash
# 1. Build the local dataset from the 10 papers
python openscholar_mini/to_litllm_schema.py \
    --papers_dir openscholar_mini/data/parsed \
    --out_dir openscholar_mini/litllm_dataset

# 2. Generate the related-works section
cd generation
PYTHONPATH=. python autoreview/models/plan_based_generation.py \
    --model_name claude-3-5-sonnet-20241022 \
    --dataset_name ../openscholar_mini/litllm_dataset \
    --gen_type vanilla \
    --prompt_type vanilla_template \
    --savedir ./outputs
```

`--model_name` also accepts `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`, or `claude-sonnet-4-20250514`.

## Note on evaluation

Since the dataset row has no gold `related_work` text (by design), the automatic ROUGE-scoring step at the end of `main()` may produce meaningless scores against an empty reference. The generated text itself is produced and printed before that step runs.
