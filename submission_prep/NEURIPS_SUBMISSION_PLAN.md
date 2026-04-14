# NeurIPS Submission Plan

## Main Benchmark

- Primary benchmark: `AV-Deepfake1M -> FakeAVCeleb NPZ-matched`
- Evaluation protocol: frozen AV-HuBERT, matched FakeAVCeleb test split, `N=1114`
- Reporting rule: seeds `{42,43,44}`, report `mean±std`
- Current primary numbers:
  - `A2`: overall `0.7720±0.0216`, FV-RA `0.5004±0.0473`
  - `A5`: overall `0.8215±0.0068`, FV-RA `0.6130±0.0143`
  - `A6`: overall `0.8439±0.0356`, FV-RA `0.6585±0.0786`

Why this should be the paper benchmark:

- It is the only protocol already aligned with the multi-seed reruns.
- The mechanistic analyses in the meeting follow-up all point back to this model family.
- It avoids mixing older `N=4089` full-FAVC runs with the newer matched-split runs.

## Paper Model Lineup

- Main text models: `A2`, `A5`, `A6`
- Main ablation/control: `A3`
- Supporting table: `A2_e2e`, `A6_e2e`
- Appendix-only controls: `A42`, `A43`, `A44`, `B1b`
- Do not use as central paper evidence: `A4`, `B1`, `A7`, `A8`, `A9`, modality-origin/manipulation sanity probes

## Main Claim To Keep

- The paper should frame the gain as `factorization changes representation usage and downstream routing`.
- Do not frame the result as true domain invariance.
- The head-ablation and subspace-probe results are the strongest evidence for the paper story.

## Must-Finish Before Submission

1. Use `mean±std` for the main benchmark table instead of single-seed values.
2. Lock the leakage-control narrative:
   - either keep `A6_leakage_trainreal` as the anti-leakage control,
   - or explicitly explain why `A6_trainvalreal` is a separate, more permissive setting.
3. Keep benchmark protocols separated in the writing:
   - matched frozen FAVC `N=1114`
   - older full-FAVC `N=4089`
   - balanced e2e `500+500`
4. Add a short benchmark comparison paragraph pointing out that current internal results are on a custom matched protocol, not the public `FakeAVCeleb 70-30` leaderboard.

## Best Single Extra Public Benchmark

- If only one additional public benchmark can be added, choose `FakeAVCeleb 70-30 split`.
- Reason: it has the richest directly comparable literature table.
- Public methods already listed in `published_fakeavceleb_sota.csv`:
  - `Xception`
  - `RealForensics`
  - `MDS`
  - `VFD`
  - `AVoiD-DF`
  - `AVFF`
- Keep `AVT2-DWF` separate because it uses a balanced FakeAVCeleb protocol, not the same 70-30 split.

## Suggested Table Layout

- Table 1: main benchmark (`A2`, `A5`, `A6`, plus `A3` as the key control), all as `mean±std`
- Table 2: mechanistic summary from probe/head-ablation results
- Appendix Table A: `A42/A43/A44`, `A2_e2e/A6_e2e`, `B1b`
- Appendix Table B: best-seed runs and per-category AUCs

## Practical Next Step

- Treat `submission_prep/main3_seed_results_summary.csv` as the source of truth for the main result table.
- Treat `meeting_followup_experiments_summary.csv` and `meeting_followup_experiments_metrics_long.csv` as presentation layers that should mirror those numbers exactly.
