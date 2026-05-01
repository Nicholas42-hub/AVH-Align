# Caren Supervisor Update

This note is a supervisor-facing briefing for the recent AVH-Align experiments. It is meant to support a short meeting update, a one-page email summary, or a 4-slide discussion.

## One-Line Takeaway

Under the unified matched FakeAVCeleb frozen protocol, `A6` remains the strongest and most paper-ready model. The cleanest story is that `A6` improves mainly on the visual-fake category `FV-RA`, and the strongest supporting evidence is the head-ablation result plus the corrected leading-silence control.

## Recommended Meeting Structure

Use this order in a meeting with Caren:

1. Start with the stable main result.
2. Show the minimum mechanistic evidence needed to justify the claim.
3. Separate newly finished but still variable results from the paper-safe core result.
4. End with risks and the recommended writing boundary.

## Slide 1: Stable Main Result

Suggested title:

`A6 is still the strongest model under the unified matched FakeAVCeleb protocol`

Suggested table:

| Model | Overall AUC | RV-FA | FV-RA | FV-FA |
| --- | ---: | ---: | ---: | ---: |
| A2 | `0.7720+-0.0216` | `1.0000+-0.0000` | `0.5004+-0.0473` | `0.9996+-0.0001` |
| A5 | `0.8215+-0.0068` | `0.9985+-0.0021` | `0.6130+-0.0143` | `0.9961+-0.0055` |
| A6 | `0.8439+-0.0356` | `1.0000+-0.0000` | `0.6585+-0.0786` | `0.9991+-0.0006` |

What to say:

- The main benchmark should stay `AV-Deepfake1M -> FakeAVCeleb NPZ-matched`, frozen AV-HuBERT, test `N=1114`, seeds `{42,43,44}`.
- The ranking is stable: `A6 > A5 > A2`.
- The practically important gain is on `FV-RA`, where A2 is near chance, A5 improves, and A6 improves again.
- For the core `A6` row, the current official number should still use the true seeded sweep `42/43/44`.
- Unlike `A6_trainvalreal`, there is no separate fourth seeded baseline run yet; the older unseeded `outputs_A6` directory is just another `seed43` run, so it should not be reused as a third independent seed.

Source of truth:

- `submission_prep/main3_seed_results_summary.csv`
- `submission_prep/unified_frozen_protocol_main.csv`

## Slide 2: Why We Believe the A6 Story

Suggested title:

`The gain is not just a better score; A6 uses a more appropriate visual route`

Show only the strongest two pieces of evidence.

### Evidence 1: Head ablation

For `A6`, masking `u_v` causes a large `FV-RA` drop:

- baseline `FV-RA = 0.7206`
- `mask_uv -> FV-RA = 0.3661`
- delta `= -0.3544`

Interpretation:

- `A6` relies on a dedicated visual pathway for the visual-fake category.
- This is the strongest mechanism result and should be one of the main paper figures.

### Evidence 2: Corrected leading-silence control

Under the corrected true-trimmed protocol:

- `A5`: `FV-RA 0.6156 -> 0.6058`
- `A6`: `FV-RA 0.7568 -> 0.7648`

Interpretation:

- The visual-fake gain does not disappear after removing raw-waveform leading silence.
- The remaining drop is concentrated in audio-fake categories, not the visual-fake category.

Recommended sentence:

`A6 is not only better on the matched benchmark; we also have direct evidence that it routes the visual-fake decision through the expected visual subspace, and that this route survives the corrected leading-silence control.`

Source of truth:

- `meeting_followup_experiments_summary.csv`
- `submission_prep/leading_silence_true_trim_results.csv`
- `submission_prep/LEADING_SILENCE_RESULTS.md`

## Slide 3: New Results To Present Carefully

Suggested title:

`Two recent follow-ups are promising, but they should not replace the stable main result yet`

### A6 train-real variants

Current 3-seed summaries:

| Variant | Overall AUC | FV-RA |
| --- | ---: | ---: |
| A6 baseline | `0.8439+-0.0356` | `0.6585+-0.0786` |
| A6_sharedonly_trainreal | `0.8516+-0.0166` | `0.6746+-0.0365` |
| A6_lite_trainreal | `0.8401+-0.0639` | `0.6503+-0.1406` |

Additional note:

- `A6_trainvalreal` is the strongest recent variant.
- For discussion with Caren, use the strongest stable trio `43,44,45`: overall `0.9039+-0.0231`, `FV-RA 0.7897+-0.0508`.
- The earlier unseeded directory `outputs_A6_trainvalreal` is also a `seed43` run.
- Treat `seed42` as an outlier under investigation rather than the default reporting set for this variant.

How to frame it:

- `trainvalreal` looks genuinely promising.
- The `43/44/45` trio is both strong and relatively tight.
- `seed42` is substantially weaker and should be mentioned as an outlier we are still diagnosing.
- So this result is best presented as a promising follow-up direction rather than the main table replacement.

### AV-Lips cross-dataset

Recent 3-seed full-head AUC means:

| Model | Full AUC mean | Seed values |
| --- | ---: | --- |
| A2 | `0.5326+-0.0315` | `0.5677, 0.5233, 0.5067` |
| A5 | `0.5880+-0.0079` | `0.5889, 0.5797, 0.5954` |
| A6 | `0.6347+-0.0876` | `0.5683, 0.7340, 0.6019` |

How to frame it:

- The direction is encouraging because the mean ordering is still `A6 > A5 > A2`.
- But `A6` is much less stable here than on matched FakeAVCeleb.
- This is supportive evidence, not a main-claim benchmark yet.

Source of truth:

- `submission_prep/a6_variant_seed_results_summary.csv`
- `avh_sup/outputs_A2_seed42/results_avlips/eval_results.txt`
- `avh_sup/outputs_A2_seed43/results_avlips/eval_results.txt`
- `avh_sup/outputs_A2_seed44/results_avlips/eval_results.txt`
- `avh_sup/outputs_A5_seed42/results_avlips/eval_results.txt`
- `avh_sup/outputs_A5_seed43/results_avlips/eval_results.txt`
- `avh_sup/outputs_A5_seed44/results_avlips/eval_results.txt`
- `avh_sup/outputs_A6_seed42/results_avlips/eval_results.txt`
- `avh_sup/outputs_A6_seed43/results_avlips/eval_results.txt`
- `avh_sup/outputs_A6_seed44/results_avlips/eval_results.txt`

## Slide 4: Risks And Recommendation

Suggested title:

`What we should claim now vs what still needs caution`

Safe claims:

- `A6 is the strongest model on the unified matched FakeAVCeleb frozen protocol.`
- `The main gain is in FV-RA, not in the already-saturated audio-fake categories.`
- `A6 shows evidence of dedicated visual routing through the expected subspace.`
- `The corrected leading-silence analysis does not support the claim that A6's FV-RA gain is a leading-silence artifact.`

Claims to avoid:

- `A6 is domain-invariant.`
- `A6 is fully robust on every cross-dataset benchmark.`
- `A6_trainvalreal is already a fully validated replacement for the paper main result.`
- `AV-Lips is already stable enough to be a central table.`

Recommendation to Caren:

- Keep the paper center on the matched FakeAVCeleb multi-seed result.
- Use head ablation plus corrected leading-silence as the main supporting evidence.
- Mention `A6_trainvalreal` as a strong follow-up using seeds `43/44/45`, while noting `seed42` as an outlier under investigation.
- Keep `AV-Lips` in reserve unless we add either more seeds or a stronger stability explanation.
- Of the newest April 8-9 results, the corrected leading-silence run is the one that most directly strengthens the main claim; the A6 train-real variants are discussion / rebuttal material, and `AV-Lips` is supportive but not central.

## Ready-To-Say 30-Second Summary

`The most stable result is still the unified matched FakeAVCeleb benchmark, where A6 is best across the three core models and the gain is mainly on FV-RA. The strongest evidence is not just the score gap, but the mechanism: the A6 head-ablation result shows it really routes the visual-fake decision through a dedicated visual subspace, and the corrected leading-silence experiment shows that this visual route survives after removing the raw-waveform silence artifact. We also finished two newer follow-ups: A6_trainvalreal looks even stronger if we use the best stable trio of seeds 43, 44, and 45, while seed42 currently behaves like an outlier under investigation; AV-Lips points in the same direction but is not yet stable enough to make it a central claim.` 

## If Caren Asks "What Should Go In The Paper?"

Short answer:

- Main table: `A2 / A5 / A6` on matched FakeAVCeleb as `mean+-std`
- Main mechanism figure: `A6` head ablation
- Supporting control: corrected leading-silence result
- Appendix or verbal update only: `A6_trainvalreal`, `AV-Lips`
