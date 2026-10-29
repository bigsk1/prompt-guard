# Evaluation corpus

`corpus/` is a seed set, not a held-out acceptance test. `status_page_compact.txt` is the reviewed false positive: ordinary status-page history in compact JSON. The pretty-printed JSON and the page markdown are the same capture stored as other raw strings. The attack files are synthetic.

The service still scores every file as raw text. Nothing here changes that path.

Run the scored report only when the pinned model is already cached:

```bash
PROMPT_GUARD_RUN_EVAL=1 python scripts/evaluate_corpus.py --cache /path/to/hf-cache
```

The script refuses to invent scores when the model is unavailable.
