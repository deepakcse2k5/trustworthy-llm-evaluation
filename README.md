# Trustworthy LLM Evaluation: companion code

Code for the book **Trustworthy LLM Evaluation: Statistics and experiment design for LLM
applications and agents** by Deepak Mishra.

The repository is a small Python package, `evidence`, built up one chapter at a time, plus one
script per chapter that prints every number quoted in that chapter.

> **All data here is simulated.** The support assistant, its versions, its traffic, the
> judges and the raters are simulations with known true values, so each method can be checked
> against the truth. The numbers describe the simulation, not any real model. The methods carry
> over to your own system; measure your own numbers.

## Setup

Python 3.11 or later.

```bash
git clone https://github.com/deepakcse2k5/trustworthy-llm-evaluation.git
cd trustworthy-llm-evaluation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                        # 85 tests should pass
```

Run the chapter scripts from the repository root, for example:

```bash
python scripts/ch04_intervals.py
```

## Chapters

| Chapter | Script | Main module(s) |
|---|---|---|
| 1. Why LLM evaluation goes wrong | `ch01_first_eval.py` | `dataset`, `simulate`, `score` |
| 2. From decision to metric | `ch02_decision_metrics.py` | `spec`, `checks` |
| 3. Building an evaluation set | `ch03_build_eval_set.py` | `traffic`, `sampling`, `leakage` |
| 4. How sure is a score? | `ch04_intervals.py` | `intervals` |
| 5. Noise you didn't plan for | `ch05_noise.py` | `noise` |
| 6. Comparing two versions | `ch06_compare.py` | `compare` |
| 7. How many examples do you need? | `ch07_sample_size.py` | `power` |
| 8. Many metrics, many slices | `ch08_slices.py` | `slices` |
| 9. A model as judge | `ch09_judge.py` | `judge` |
| 10. Human labels you can rely on | `ch10_labels.py` | `labels` |
| 11. Evaluating RAG and agents | `ch11_rag_agents.py` | `rag`, `agents` |
| 12. Evidence in production | `ch12_production.py` | `production` |
| 13. Reporting evidence | `ch13_report.py` | `report` |

Scripts are deterministic: each uses fixed random seeds, so its output matches the book.
A few (Chapters 5, 8, 12) run thousands of simulated evaluations and take up to a minute.

## Command line

```bash
# compare two versions run on the same questions (Chapter 6)
python -m evidence compare data/ch06_v1.json data/ch06_v2.json \
    --items data/ch06_items.jsonl --labels v1 v2

# plan an evaluation (Chapter 7)
python -m evidence size rate --rate 0.8 --half-width 0.03
python -m evidence size paired --discordant 0.09 --effect 0.03
python -m evidence size detectable --questions 1000 --discordant 0.25
```

The comparison files are written by `scripts/ch06_compare.py`; run it first.

## Using it on your own system

The functions take plain Python data: dictionaries mapping a question id to 1 (pass) or 0
(fail), and lists of `Item` records with an id and a category. Replace the simulated runs with
your system's results in that shape, and the intervals, comparisons, sample-size plans, judge
calibration and reports work unchanged. Start with `evidence.spec` and write your own metric
specification before running anything (Chapter 2).

## License

MIT. See `LICENSE`.
