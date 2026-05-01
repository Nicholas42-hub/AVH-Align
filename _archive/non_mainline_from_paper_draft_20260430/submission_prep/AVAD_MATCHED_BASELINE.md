# AVAD Matched Baseline

Goal: add one external baseline that is closer to our AV1M -> FAVC protocol than
the published LRS-trained AVAD row reproduced from AVH-Align.

## Why AVAD First

- AVAD is already cited and already appears in the broader protocol-context
  table.
- The official code exposes a simple `detect.py` interface that consumes a text
  file of mp4 paths and writes `testing_scores.npy`.
- A FAVC 70/30 category evaluation can be run immediately with our existing
  raw FakeAVCeleb paths.
- A matched AV1M-trained variant would directly address the "only one external
  supervised baseline" critique.

## Files Added

- `submission_prep/prepare_avad_protocol.py`
  - `prepare`: resolves the FAVC 70/30 test CSV to absolute raw-video paths and
    writes AVAD's input list plus label metadata.
  - `evaluate`: reads AVAD's `testing_scores.npy` and reports Overall, RV-FA,
    FV-RA, and FV-FA AUC/AP under our category protocol.
- `run_avad_favc7030_eval.slurm`
  - clones `cfeng16/audio-visual-forensics` if absent,
  - prepares the FAVC test list,
  - runs official AVAD `detect.py`,
  - converts AVAD scores to our category table.

## First Run

```bash
sbatch run_avad_favc7030_eval.slurm
```

The job expects the official AVAD synchronization checkpoint at:

```text
external/audio-visual-forensics/sync_model.pth
```

The README links this checkpoint through Google Drive, so it may need to be
downloaded manually on Spartan before rerunning the job.

## Reporting Rule

Use the first successful result only as:

```text
AVAD official checkpoint, evaluated on our FAVC 70/30 category protocol
```

Do not describe it as "AV1M-supervised" unless the AVAD model is actually
retrained using AV1M data.

## AV1M-Trained Variant

The stronger paper result would be:

```text
AVAD retrained on AV1M source data, evaluated on FAVC 70/30
```

That requires one extra implementation decision because the public AVAD README
mainly documents inference rather than a complete training recipe. The cleanest
version is to train the same anomaly model on AV1M real clips only, then evaluate
all FAVC categories. If time is too tight, the official-checkpoint evaluation is
still useful as external context, but it does not fully remove the matched
external-baseline concern.
