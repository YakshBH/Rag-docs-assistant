# Evaluation results

Do not enter invented metrics. Run the evaluation script and paste the actual output here.

| Version | Change | Recall@5 (source-page hit) | MRR | Notes |
|---|---|---:|---:|---|
| Reported baseline | Original semantic-search baseline | 0.675 | 0.397 | 40 questions, as supplied |
| Before changes | Existing dense + substring keyword fusion | 0.650 | 0.412 | 26/40, measured with `eval/run_eval.py` |
| Dense + BM25, page-level RRF | Dense + BM25, page-level RRF | 0.700 | 0.461 | 28/40, measured after rebuilding the index |
| Current | Phrase-aware page fusion | 0.725 | 0.470 | 29/40, measured with `eval/run_eval.py` |

## Failure analysis
For each failure, record the question, expected source page, retrieved pages, likely cause, and the next change you will test.

The evaluation labels were verified against the extracted corpus: every expected
`source`/`page` pair in `eval/questions.json` has a corresponding
`data/processed/<source stem>_page_<page>.txt` file. The rebuild extracted 178
non-empty pages from the four PDFs and indexed 218 chunks. The current evaluator
still uses exact source-page matching and does not alter labels.
