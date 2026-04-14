# Leading Silence Results

This note records the corrected paper-facing interpretation of the leading-silence experiment.

## Correct Protocol

- Deprecated diagnostic run: `outputs_A5_seed43/results_silence_rob` and `outputs_A6_seed43/results_silence_rob`
  These runs zeroed the first `K` AV-HuBERT audio feature frames and should be treated only as a stress test.
- Correct paper-facing run: `outputs_A5_seed43/results_silence_rob_trimmed` and `outputs_A6_seed43/results_silence_rob_trimmed`
  These use `data/favc_features_trimmed`, where leading silence was removed from the raw waveform before AV-HuBERT feature extraction.

## Corrected Takeaway

- A5 true-trimmed: overall `0.8186 -> 0.7489`, but `FV-RA` stays nearly unchanged: `0.6156 -> 0.6058`.
- A6 true-trimmed: overall `0.8885 -> 0.8187`, while `FV-RA` is stable to slightly higher: `0.7568 -> 0.7648`.
- The post-trim drop is concentrated in the audio-fake categories (`RV-FA` / `FV-FA`), not in the visual-fake category (`FV-RA`).

## Paper-Ready Description

Suggested short version:

> We re-ran the leading-silence analysis with a corrected protocol that removes the raw-audio prefix silence before re-extracting AV-HuBERT features. Under this true-trimmed evaluation, the visual-fake category `FV-RA` remains stable for both A5 (`0.6156 -> 0.6058`) and A6 (`0.7568 -> 0.7648`), while the overall drop is driven by the audio-fake categories. This indicates that A6's visual OOD pathway is not dependent on the leading-silence artifact, even though audio-related performance remains sensitive to prefix-level audio distribution changes.

Suggested slightly longer version:

> The initial feature-prefix zeroing experiment overstated the effect of leading silence because it perturbed already-extracted AV-HuBERT features rather than removing the silence from the raw waveform. Using the corrected true-trimmed features, we find that A6 preserves `FV-RA` performance (`0.7568 -> 0.7648`) and A5 also remains nearly unchanged on `FV-RA` (`0.6156 -> 0.6058`). The residual drop appears in `RV-FA` and `FV-FA`, suggesting sensitivity in the audio-fake pathway rather than a collapse of the visual-fake route.

## Main-Text Paragraph

Recommended final wording for the paper body:

> We revisited the leading-silence analysis with a corrected protocol that removes the raw-audio prefix silence before re-extracting AV-HuBERT features, rather than zeroing feature prefixes post hoc. Under this true-trimmed evaluation, the visual-fake category remains stable: A5 changes only from `0.6156` to `0.6058` on `FV-RA`, while A6 slightly improves from `0.7568` to `0.7648`. The remaining performance drop is concentrated in the audio-fake categories (`RV-FA` and `FV-FA`), indicating that the main visual OOD detection pathway learned by A6 does not depend on the leading-silence artifact, although audio-fake performance is still sensitive to prefix-level audio distribution shifts.

## Appendix Paragraph

Recommended appendix wording:

> We initially evaluated leading-silence robustness by zeroing the first `K` AV-HuBERT audio feature frames. We no longer use that result as the paper-facing conclusion because it perturbs already-extracted features instead of removing silence from the raw waveform. In the corrected experiment, we evaluate the same A5/A6 checkpoints on `favc_features_trimmed`, where leading silence is removed before AV-HuBERT extraction. With this protocol, `FV-RA` remains stable for both models (A5: `0.6156 -> 0.6058`, A6: `0.7568 -> 0.7648`), whereas the drop appears mainly in `RV-FA` and `FV-FA`. This supports the claim that A6's visual-fake route is not driven by leading silence, while clarifying that audio-related decisions remain distribution-sensitive.

## Table Caption

Recommended caption for a supplementary table:

> Corrected leading-silence evaluation using true-trimmed FakeAVCeleb features. Unlike the earlier diagnostic prefix-zeroing stress test, this protocol removes leading silence from the raw waveform before AV-HuBERT feature extraction. `FV-RA` remains stable after trimming, especially for A6, while most of the degradation appears in the audio-fake categories.

## Claim-Safe Interpretation

Use these claim boundaries consistently:

- Safe: `A6's visual OOD pathway is robust to leading-silence removal.`
- Safe: `The corrected trimming analysis does not support the hypothesis that A6's FV-RA gains come from the leading-silence artifact.`
- Safe: `Audio-fake categories remain sensitive to prefix-level audio distribution changes.`
- Avoid: `A6 is fully invariant to leading silence.`
- Avoid: `The leading-silence artifact has no effect on the model at all.`
