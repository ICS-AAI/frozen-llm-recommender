# frozen-llm-recommender

A Python harness and dataset for evaluating frozen large language models (LLMs) as recommenders of machine-learning
pipelines. The harness gives an LLM the anonymized meta-features of a dataset and a catalog of candidate pipelines (OpenML
flows), asks it to rank the candidates, parses the returned ranking, and logs every attempt it makes (request and response
ids, served model, stop reason, token usage). It talks to Anthropic models and to OpenAI-compatible endpoints (OpenAI,
DeepInfra), sends the shared prompt prefix (instructions and catalog) so that it can be cached (explicitly on Anthropic,
automatically on OpenAI), asks up to three times when the answer is empty or truncated, and can stop a run at a spending
limit. Scoring functions compute normalized regret of a ranking against held-out accuracies, map flows to estimator
families, and build leave-one-task-out family portfolios. The repository ships the rankings, ground truth and API logs of
the study in the paper "Frozen LLMs as Recommenders for AutoML Pipeline Selection: Can We Trust the Evaluation?" (TAE
workshop at NeurIPS 2026), and scripts that reproduce Tables 4 and 5 of that paper and the point estimates and p-values of
Tables 1 and 2 from them without API keys or OpenML access. It is meant for researchers in AutoML and meta-learning who want to check or extend that evaluation.
Reproducing the tables needs Python and the packages in `requirements.txt`; new model runs need an API key and a task
pack, which are not included.

Repository: https://github.com/ICS-AAI/frozen-llm-recommender (the repository address printed in the TAE paper redirects here).

## Installation

```bash
git clone https://github.com/ICS-AAI/frozen-llm-recommender.git
cd frozen-llm-recommender
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13.5. `requirements.txt` lists numpy, pandas, scipy, scikit-learn, statsmodels, joblib, anthropic and
openai (unpinned). The harness was checked only with stand-in API clients (no request was sent); it has not been run
against the APIs with the package versions installed here. Run the commands of the next three sections from the
repository root.

There is no `pyproject.toml`: the scripts read the shipped data from `results/` in the clone, and the harness resolves its
default input and output paths from the repository location, so an installed copy (outside the clone) would not find them.
Use the clone.

## Quick start (no API key)

```bash
python reproduce_tables.py    # Table 5 and the point estimates and p-values of Tables 1 and 2
python table4.py              # Table 4, the Llama-Turbo control and the Appendix A determinism check
```

Both read only `results/` and print to the terminal.

## Usage

Scoring and family functions (`table4.py`; importing it runs nothing):

```python
import table4 as t
truth = t.load_truth("results/cc18/ground_truth_cc18.csv")       # task -> candidates with accuracies and families
rankings = t.load_rankings("results/cc18/rankings_cc18.csv")     # model -> task -> ranked candidate ids
model = "claude-haiku-4-5-20251001"
scores = [t.nreg_flow(r, truth[k]) for k, r in rankings[model].items() if k in truth]   # nregret@1 per task
scores = [s for s in scores if s is not None]                    # None: all candidates tie, or no ranked id is a candidate
print(model, len(scores), round(sum(scores) / len(scores), 3))
```

Other functions in `table4.py`: `family(flow_name)` (flow name -> estimator family by regular expressions, else "other"),
`nreg_fam` and `fam_order` (family-level regret and order), `family_portfolios(truth)` (leave-one-task-out family
portfolios ordered by mean member accuracy (a) or by best member (b)), `bh(p)` (Benjamini–Hochberg), `compare` (Wilcoxon
signed-rank against a baseline, with BH), `boot_ci`, `paired`. `load_truth` needs a `flow_name` column; for
`results/ground_truth.csv`, which has none, pass the catalog:

```python
import pandas as pd, table4 as t
names = pd.read_csv("results/catalog.csv")[["candidate_id", "flow_name"]]
truth = t.load_truth("results/ground_truth.csv", names)          # the 200 long-tail tasks
print(len(truth))
```

```bash
python -c "import table4 as t; print(t.family('sklearn.ensemble.forest.RandomForestClassifier(13)'))"   # random_forest
```

Harness helpers (`src/llm_paper4.py`; importing it creates no output directory):

```bash
python -c "import sys; sys.path.insert(0, 'src'); from llm_paper4 import _parse_ranking; print(_parse_ranking('F3, F1, F9, F3', {'F1', 'F3', 'F7'}))"
# ['F3', 'F1']: valid ids in order of first appearance
python -c "import sys; sys.path.insert(0, 'src'); from llm_paper4 import _provider_and_model as p; print(p('gpt-5-mini'), p('deepinfra:deepseek-ai/DeepSeek-V3'), p('claude-haiku-4-5-20251001'))"
# ('openai', 'gpt-5-mini') ('deepinfra', 'deepseek-ai/DeepSeek-V3') ('anthropic', 'claude-haiku-4-5-20251001')
```

### Commands that need inputs not included in this repository

`build` writes a task pack (`pack.joblib`) from the authors' OpenML cache; `dryrun`, `run` and `eval` read a pack; new runs
need an API key. The cache, the packs and keys are not shipped.

| command | needs | where it comes from |
|---|---|---|
| `python src/llm_paper4.py build` | `data/minimal_cache_cc18/longtail_collected.joblib` and `data/minimal_cache_cc18/longtail_tasks.joblib` | the authors' OpenML long-tail cache, collected from the OpenML API by code that is not part of this repository |
| `P4_PACK=<pack.joblib> python src/llm_paper4.py dryrun` | a task pack | written by `build`; without `P4_PACK` the default `results/results_recommender_paper4/pack.joblib`, which is built first (and so needs the cache) |
| `P4_PACK=<pack.joblib> P4_OUT=<dir> python src/llm_paper4.py run <model> [N]` | a task pack and the provider's API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `DEEPINFRA_API_KEY`, or `~/.anthropic_key`, `~/.openai_key`, `~/.deepinfra_key`) | the pack as above; the key from the provider |
| `python src/llm_paper4.py eval` | the pack, the run outputs and `results/results_recommender_longtail_full/ranking_metrics_per_task.csv` | the authors' trained-baseline run (not shipped) |
| `python src/llm_sig.py` | `results_recommender_paper4/llm_metrics_per_task.csv` and `results_recommender_longtail_full/ranking_metrics_per_task.csv`, relative to the working directory | the authors' working directories (not shipped in that layout) |
| `src/llm_pack_cc18_p4.py`, `src/llm_pack_turbo50.py`, `src/run_cc18_rerun.py`, `src/score_cc18_rerun.py` | the authors' `results/` and `data/` layout (CC18 OpenML cache, packs) | not shipped; `table4.py` is the self-contained equivalent of the scoring |

A task pack is a joblib file holding a dict: `prefix` (instructions + catalog, sent as the cached system prompt), `packs` (a
list of `{"tid", "suffix", "cids"}`: task id, per-task prompt with the meta-features and candidate ids, candidate ids),
`truth` (task id -> list of `{"flow_id", "acc", "cat"}`) and `fid2cat` (flow id -> candidate id). `dryrun` prints a token
and cost estimate per model for a pack without a key (prices in `PRICING` are approximate). `run` writes, under `P4_OUT`,
`llm_<model>.joblib` (rankings and attempts per task), `responses.jsonl` (raw answer texts) and `api_log.jsonl`: one line
per attempt with `tid`, `attempt`, `ts`, `model`, `request_id`, `response_id`, `resp_model`, `stop`, `truncated`, `inp`,
`out`, `cw`, `cr` (input, output, cache-write and cache-read tokens) and `n_parsed`, or `error` for a failed request.
`P4_BUDGET_DIRS` (colon-separated output directories) and `P4_BUDGET_STOP` (dollars) stop a run before a new task once the
logged spend of those directories reaches the limit.

## Data shipped with the package

`results/` holds 27 files, 11,561,792 bytes in total: 21 CSV tables, 5 JSON-lines files (3 API logs and 2 re-ask
archives) and 1 Markdown score report. They come from the 200-task long-tail study (with the leakage ablation and the
seed-123 replication), the August 2026 DeepSeek-V3 run and the October 2026 re-run on the OpenML-CC18 tasks, the
all-toolkit CC18 pack, and the 50-task Llama-Turbo control.
Formats and provenance are listed under "What's here" in the next section.

## Reproducing the paper (TAE workshop at NeurIPS 2026)

Code and data for the paper

> Roman Neruda, Juan Carlos Figueroa-García, Carlos Alberto Franco Franco.
> *Frozen LLMs as Recommenders for AutoML Pipeline Selection: Can We Trust the Evaluation?*
> TAE (Trust-AI-Eval): Can We Trust AI Evaluation? — workshop at NeurIPS 2026.

This repository lets anyone **reproduce Tables 4 and 5 and the point estimates and p-values of Tables 1 and 2 from the
released CSVs with no API keys and no OpenML access**. New runs of the evaluation need inputs that are not included (see
"Commands that need inputs not included in this repository").

### Quick reproduction (no API, ~10 s)

```bash
pip install -r requirements.txt
python reproduce_tables.py
python table4.py              # Table 4 (CC18 and long tail, flow and family level), Turbo control, Appendix A
```

`reproduce_tables.py` prints Table 1 (per-model quality), Table 2 (mean-difference vs. portfolio,
BH-corrected — 10/11 worse) and Table 5 (task-invariance / degeneracy of top-1 picks). `table4.py` prints
Table 4 with and without the "other" family bucket, the Llama-Turbo (FP8) control and the DeepSeek-V3
serving-determinism check of Appendix A. Both read only `results/`.

### What's here

```
reproduce_tables.py          reproduce Table 5 and the point estimates and p-values of Tables 1, 2 from the CSVs
table4.py                    reproduce Table 4 + Turbo control + Appendix A determinism check
requirements.txt
src/
  llm_paper4.py              evaluation harness (build | dryrun | run <model> [N] | eval); per-call API log,
                             up to 3 attempts for empty/truncated answers, spend guard
  llm_sig.py                 paired significance, mixed-effects, BH correction, tier tests
  llm_pack_cc18_p4.py        CC18 task packs for the harness (same instructions / meta-features / format)
  llm_pack_turbo50.py        50-task long-tail pack for the Llama-Turbo control
  run_cc18_rerun.py          resumable driver of the October 2026 CC18 re-run
  score_cc18_rerun.py        scoring used for the re-run (reads the authors' internal directory layout;
                             table4.py is the self-contained equivalent on the released CSVs)
results/
  ground_truth.csv           (task_id, candidate_id, accuracy) — the held-out labels, self-contained
  catalog.csv                (candidate_id, flow_id, flow_name) — candidate IDs mapped to OpenML flows
  rankings_main200.csv       per-model top-to-bottom rankings, 11 models x 200 tasks (+ token usage)
  llm_metrics_per_task.csv   per-task nregret@1 / hit@1 / Spearman, per model
  baselines_main200.csv      trained text-LTR / kNN / portfolio / random, per task (200-subset)
  rankings_leakage200_*.csv  leakage ablation (dataset name revealed), Sonnet & DeepSeek, n=200
  rankings_seed123pt_*.csv   independent second seed (stability), per-task catalog
  cc18/                      October 2026 CC18 re-run (69 tasks with >= 15 scikit-learn candidates)
    ground_truth_cc18.csv    (task_id, candidate_id, flow_id, flow_name, accuracy)
    rankings_cc18.csv        11 models x 69 tasks: ranking, coverage, truncation, chosen attempt, tokens
    portfolio_cc18.csv       per-task nregret@1 of the flow portfolio (trained without the evaluated task)
    api_log_cc18.jsonl       one line per API call: request/response id, served model, stop reason,
                             token usage, timestamp, number of parsed ids
    reask_archive_cc18.jsonl earlier attempts of the tasks that were re-asked (see below)
    rankings_cc18_deepseek_aug2026.csv  the August 2026 DeepSeek-V3 run (no API log), for Appendix A
    request_config_cc18.csv  request id -> configuration of every attempt behind a released ranking
  cc18_all/                  supplementary all-toolkit CC18 pack (72 tasks), mentioned in Sec. 4.3, not in Table 4
    ground_truth_cc18_all.csv, rankings_cc18_all.csv (8 models x 72 tasks), portfolio_cc18_all.csv,
    api_log_cc18_all.jsonl, reask_archive_cc18_all.jsonl, request_config_cc18_all.csv (same formats as cc18/)
    score_cc18_all.md        output of src/score_cc18_rerun.py for this pack
  turbo50/                   Llama-Turbo (FP8) control on 50 long-tail tasks: rankings, task list, API log
```

### CC18 re-run (October 2026) and changes to Table 4

The CC18 column of Table 4 comes from a re-run on 27 Oct 2026 of all eleven configurations with this
harness and prompt on the 69 OpenML-CC18 tasks with at least 15 scikit-learn candidates (catalog 1,475
flows). Every call is logged in `results/cc18/api_log_cc18.jsonl`. Details:

- Llama-3.3-70B and Llama-3.1-8B are no longer served under their original ids; the re-run uses the FP8
  "-Turbo" variants. A 50-task long-tail control (`results/turbo50/`) compares them with the original runs:
  70B matches (paired difference -0.006, 95 % CI [-0.048, +0.031]), 8B is somewhat worse (+0.066,
  [-0.009, +0.140]).
- Each task is asked up to three times if the answer is empty or truncated. A normally finished answer with
  a ranking is used; otherwise the attempt with the longest valid ranking (selection by length only). For
  nine tasks whose longer partial ranking had been lost by an earlier version of the harness, the task was
  asked again; the earlier attempts are kept in `reask_archive_cc18.jsonl`. (The 200-task main study used one
  attempt per task.)
- Portfolios are computed without the evaluated task: the flow portfolio of the main study (task-grouped CV)
  and two leave-one-task-out family portfolios, (a) by mean member accuracy and (b) by best member.

Changes relative to the submitted version of the paper:

- **Flow / CC18 cell.** The submitted value (DeepSeek-V3 0.141 vs. portfolio 0.070) used a portfolio that
  averaged flow accuracies over all CC18 tasks *including the evaluated task*. With the leakage-free
  portfolio of the main study the portfolio scores 0.090. The DeepSeek-V3 rankings also changed between the
  August run and the logged October re-run (identical prompt, temperature 0; top-1 differs on 24 of 69 tasks);
  Table 4 uses the logged run (0.086). Both runs are released.
- **Family / CC18 cell.** The submitted value (0.060 vs. 0.095) did not come from an LLM API call: the
  per-task family scores were written by the AI coding assistant used in the project, acting as the "frozen
  LLM" on the anonymized meta-features. The files are kept, marked as quarantined, in the authors' working
  repository and are not used anywhere; the cell is replaced by the API re-run above.
- **Family / long-tail cell.** The earlier `family_level.py` ordered the family portfolio by mean accuracy
  over all 200 tasks, including the evaluated one. `table4.py` uses leave-one-task-out portfolios (a) and (b)
  on the common 161-task set.
- **Family baseline.** With portfolio (a) the LLMs appear to win at the family level; with portfolio (b) no
  model wins on either benchmark. Results without the "other" family bucket are printed as well.

- **API log labels.** In `api_log_cc18.jsonl` (and for part of `api_log_cc18_all.jsonl`) calls of the
  Sonnet 5 + thinking configuration are logged under the base id `claude-sonnet-5`; the harness was fixed
  afterwards. `request_config_*.csv` maps every request id behind a released ranking to its configuration
  (every such request id is present in the API log). API-log lines with no entry in that map are earlier
  attempts whose results were superseded when the task was asked again, or requests that failed.

### Supplementary all-toolkit CC18 pack (Sec. 4.3)

The same harness, prompt and retry rule on all 72 OpenML-CC18 tasks with candidates from all toolkits
(catalog 3,073 flows; `src/llm_pack_cc18_p4.py`, pack A). Eight configurations were run: DeepSeek-V3 and
Qwen2.5-72B are served with a 32,768-token limit, below this prompt (DeepSeek-V3 calls returned HTTP 400 and
are in the API log; Qwen2.5-72B was not run), and Claude Fable 5 ended every attempt at the output-token limit with an
empty ranking (stopped after 8 attempts). Sonnet 5 + thinking gave no ranking on one task (3903), so the
common set has 71 tasks. On it no model is significantly better than the flow portfolio (0.090) and 4 of 8 are
significantly worse (BH q < 0.05); no model is significantly better than family portfolio (a). Full output:
`results/cc18_all/score_cc18_all.md`. This pack is not part of Table 4, and `table4.py` does not read it.

### Re-running the LLM evaluation (needs API keys and inputs not included here)

These commands need the inputs listed under "Commands that need inputs not included in this repository" (the OpenML
long-tail cache for `build`, a task pack and an API key for `run`, the trained-baseline file for `eval`).

The harness infers the provider from the model id and reads each provider's key from an env var or
`~/.<provider>_key` (Anthropic / OpenAI / DeepInfra). The prompt (instructions + flow catalog) is cached;
open-weight models are served via an OpenAI-compatible endpoint.

```bash
cd src
python llm_paper4.py build                 # build task pack (needs the OpenML long-tail cache)
python llm_paper4.py run gpt-5             # or: run "deepinfra:deepseek-ai/DeepSeek-V3", run claude-sonnet-5, ...
python llm_paper4.py eval                  # score all runs vs the trained baselines
python llm_sig.py                          # significance tables
```

Notes: OpenAI reasoning models must run at low `reasoning_effort` (default effort does not reliably
terminate on large-candidate ranking prompts — see the paper's Threats section). DeepInfra does not
prefix-cache. Per-task token usage is logged (used for the effort analysis).

## Changes in 1.0.1

`src/llm_paper4.py` resolves its default paths from the repository location and creates its output directory only when it
writes; in 1.0 it created `results/results_recommender_paper4` relative to the working directory at import, so
`cd src; python llm_paper4.py build` failed at import. Prompts, parsing, scoring, model settings and outputs are unchanged.

## Author, licence and owner

Software: Roman Neruda, Institute of Computer Science of the Czech Academy of Sciences. Licence: MIT (see `LICENSE`).
Copyright holder, as in `LICENSE`: Roman Neruda, Institute of Computer Science of the Czech Academy of Sciences.

## How to cite

The software:

```bibtex
@software{neruda2026frozenllmrecommender,
  author  = {Neruda, Roman},
  title   = {frozen-llm-recommender},
  version = {1.0.1},
  year    = {2026},
  url     = {https://github.com/ICS-AAI/frozen-llm-recommender}
}
```

The paper:

```bibtex
@inproceedings{neruda2026frozenllm,
  author    = {Neruda, Roman and Figueroa-Garc{\'i}a, Juan Carlos and Franco Franco, Carlos Alberto},
  title     = {Frozen {LLMs} as Recommenders for {AutoML} Pipeline Selection: Can We Trust the Evaluation?},
  booktitle = {TAE (Trust-AI-Eval): Can We Trust AI Evaluation? Workshop at NeurIPS 2026},
  year      = {2026}
}
```

## Funding and acknowledgement

This work was supported by the Czech Academy of Sciences under the Strategy AV21 research programme *Epicentres of
Civilization – Intelligent Households, Technology and Society* (StrategieAV21/37), and by the institutional support
RVO: 67985807.
