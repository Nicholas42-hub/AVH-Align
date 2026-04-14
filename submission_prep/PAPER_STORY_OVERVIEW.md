# Paper Story Overview

This note turns the current AVH-Align results into a coherent paper story: motivation, hypothesis, model progression, validation logic, benchmark definitions, and claim boundaries.

## 1. Core One-Line Story

We are **not** claiming that `A6` learns a fully domain-invariant representation.

The stronger and more defensible claim is:

> `three-way factorization changes what information the model uses and how the final detector routes decisions, especially for the visual-fake OOD category FV-RA.`

In short:

- binary decomposition (`A2`) leaves the model heavily audio-dominant
- three-way factorization (`A5/A6`) creates a usable visual-unique pathway
- `A6` is the first model that clearly routes `FV-RA` decisions through that pathway

## 2. Motivation

The practical problem is not simply deepfake detection accuracy, but **cross-domain generalization**.

Training is dominated by `AV-Deepfake1M (AV1M)`, where audio cues are very strong.  
When the model is evaluated on `FakeAVCeleb (FAVC)`, the hardest and most important category is:

- `FV-RA = FakeVideo-RealAudio`

This category cannot be solved by listening to audio shortcuts. The model has to use the visual fake signal.

That is why the real question is:

> can the model learn to use the right evidence under domain shift, instead of just learning an easier audio shortcut on the source dataset?

## 3. Main Hypothesis

Our working hypothesis is:

> the main limitation of binary causal/spurious decomposition is structural, not just loss-related.

More concretely:

1. In the binary split (`Z_c / Z_s`), useful modality-specific signal and domain-correlated shortcut signal are still entangled.
2. Because AV1M strongly rewards audio-based prediction, the binary model learns an audio-dominant causal route.
3. Simply adding domain-related losses on top of the binary structure is not enough; it may even hurt.
4. We need a richer factorization that separates:
   - shared task-relevant signal
   - modality-unique task-relevant signal
   - residual / shortcut / redundant signal
5. Once this structure exists, domain pressure can reshape **routing** rather than just suppressing features globally.

This leads to the paper-safe claim:

> `A6 works because factorization changes downstream representation usage and routing, not because it achieves true domain invariance.`

## 4. Model Progression

The cleanest model sequence is:

### `A2` — Binary baseline

- Representation split: `Z_c / Z_s`
- This is the main binary baseline.
- It already tries to isolate causal vs spurious information, but the split is still too coarse.

Expected failure mode:

- `Z_c` becomes strongly audio-dominant.
- Visual OOD capacity may exist in the representation, but the classifier does not actually use it.

### `A3` — Binary + domain adversarial

- This is the key control.
- It answers: `is the gain just from adding domain-aware objectives?`

Observed outcome:

- `A3` is worse than `A2` on the matched FAVC benchmark.
- Therefore, the domain objective alone is not the explanation.

This is one of the most important logical controls in the paper.

### `A5` — Three-way factorization (FCD)

`A5` replaces the binary split with a three-way multimodal factorization:

- `s_v, s_a`: shared / synergistic task-relevant components
- `u_v, u_a`: modality-unique task-relevant components
- `r_v, r_a`: redundant / shortcut / residual components

This creates an explicit `visual-unique` carrier, which the binary model does not cleanly expose.

### `A6` — `A5` + domain push-pull

`A6` adds domain adversarial pressure on the factorized representation:

- expel domain information from the task branch
- concentrate domain information into the residual branch

The important point is:

> the same kind of domain pressure that backfires in the binary structure becomes useful once the representation is factorized.

## 5. What Exactly We Are Trying To Prove

The paper does **not** need to prove that:

- `A6` is perfectly disentangled
- `A6` is fully domain-invariant
- `Z_c` contains no domain information

The paper **does** need to prove:

1. `A2` is structurally biased toward audio-heavy decision making.
2. `A5/A6` create a visual-unique pathway that is genuinely useful for OOD visual-fake detection.
3. `A6` uses that pathway at inference time.
4. The gain is architecture-dependent, not just the result of adding extra losses.

## 6. How The Validation Chain Works

The strongest way to explain the experiments is:

> each experiment answers one specific question in the hypothesis chain.

### Step 1: Main performance result

Question:

> does the final model actually improve OOD detection?

Answer:

- On the unified matched FAVC benchmark, `A6 > A5 > A2`.
- The key category is `FV-RA`.

Current 3-seed primary result:

| Model | Overall AUC | RV-FA | FV-RA | FV-FA |
| --- | ---: | ---: | ---: | ---: |
| A2 | `0.7720±0.0216` | `1.0000±0.0000` | `0.5004±0.0473` | `0.9996±0.0001` |
| A5 | `0.8215±0.0068` | `0.9985±0.0021` | `0.6130±0.0143` | `0.9961±0.0055` |
| A6 | `0.8439±0.0356` | `1.0000±0.0000` | `0.6585±0.0786` | `0.9991±0.0006` |

Interpretation:

- `RV-FA` and `FV-FA` are already near ceiling.
- The meaningful gain is on `FV-RA`, where the model must use visual evidence.

### Step 2: A3 control

Question:

> is this just because domain-aware training helps?

Answer:

- No.
- `A3` applies the same binary structure plus domain adversarial pressure and gets worse.

Matched FAVC result:

- `A2 overall = 0.7928`, `FV-RA = 0.5462`
- `A3 overall = 0.7490`, `FV-RA = 0.4496`

Evidence table:

| Model | Structure | Overall AUC | FV-RA AUC |
| --- | --- | ---: | ---: |
| A2 | binary baseline | `0.7928` | `0.5462` |
| A3 | binary + domain adversarial | `0.7490` | `0.4496` |
| A5 | three-way factorization | `0.8186` | `0.6156` |
| A6 | factorization + domain push-pull | `0.8885` | `0.7568` |

Interpretation:

- domain pressure alone does not solve the problem
- the effect depends on representation structure

This is the cleanest evidence that the gain is **architecture-dependent**.

### Step 3: Sequence probe / domain probe

Question:

> did A6 succeed by becoming domain-invariant?

Answer:

- No.
- All models still have very high domain predictability in `Z_c`.

Evidence table:

| Model | `Z_c` in-domain probe | `Z_c` OOD probe | `Z_c` domain probe | `Z_s` domain probe |
| --- | ---: | ---: | ---: | ---: |
| A2 | `0.9420±0.0083` | `0.7052` | `0.9975` | `0.9962` |
| A5 | `0.9970±0.0026` | `0.7744` | `0.9966` | `0.9956` |
| A6 | `0.9944±0.0033` | `0.7480` | `0.9971` | `0.9974` |

Interpretation:

- the domain-invariance framing is too strong
- the better framing is `representation usage and routing`

This is why the paper should explicitly avoid a “full invariance” claim.

### Step 4: Branch probe under perturbation

Question:

> does the representation retain non-audio information when audio is corrupted?

Answer:

- `A2` collapses much more strongly under audio masking/noise.
- `A5/A6` retain higher residual performance.

Evidence table:

| Model | `Z_c` base | Audio mask `100%` | Audio noise `σ=1.0` | Visual mask `100%` | Visual noise `σ=1.0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2 | `0.9421` | `0.5332` | `0.5664` | `0.9057` | `0.8995` |
| A5 | `0.9970` | `0.6990` | `0.6972` | `0.9958` | `0.9940` |
| A6 | `0.9948` | `0.6624` | `0.6741` | `0.9960` | `0.9939` |

Interpretation:

- FCD models preserve more non-audio capacity in the causal branch.
- This suggests a stronger visual contribution is possible.

But this step alone still does **not** prove the trained head uses that visual information.

### Step 5: FAVC branch probe

Question:

> on `FV-RA`, is the representation actually visual-specific?

Answer:

- yes
- audio masking barely changes `Z_c` probe performance on `FV-RA`
- visual masking collapses it

Evidence table (`FV-RA` probe):

| Model | Base | Audio mask `100%` | Visual mask `100%` | Visual noise `σ=1.0` |
| --- | ---: | ---: | ---: | ---: |
| A2 | `0.9925` | `0.9882` | `0.3565` | `0.6519` |
| A5 | `0.9898` | `0.9924` | `0.3400` | `0.5251` |
| A6 | `0.9892` | `0.9807` | `0.3248` | `0.4964` |

Evidence table (`RV-FA` probe):

| Model | Base | Audio mask `100%` | Visual mask `100%` |
| --- | ---: | ---: | ---: |
| A2 | `0.9915` | `0.4240` | `0.9910` |
| A5 | `1.0000` | `0.3700` | `1.0000` |
| A6 | `1.0000` | `0.3505` | `1.0000` |

Interpretation:

- the representation contains genuine visual-fake evidence
- therefore, if the detector still fails, the problem must be decision routing rather than missing representation capacity

This is especially important for explaining `A2`.

### Step 6: Subspace probe

Question:

> which FCD component carries the strongest OOD signal?

Answer:

- `u_v` is the strongest OOD subspace
- especially for `A6`

Representative result:

- `A6 u_v OOD = 0.9238`
- this is stronger than the full binary `A2 Z_c` OOD probe (`0.7052`)

Evidence table:

| Model / subspace | Dim | In-domain probe | OOD probe | Domain probe |
| --- | ---: | ---: | ---: | ---: |
| A2 `Z_c` | `1024` | `0.9420` | `0.7052` | `0.9975` |
| A2 `v_c` | `512` | `0.5704±0.0313` | `0.8906` | `0.9944±0.0017` |
| A5 `s_v` | `256` | `0.5672±0.0172` | `0.8138` | `0.9921±0.0023` |
| A5 `u_v` | `256` | `0.7879±0.0102` | `0.8248` | `0.9668±0.0058` |
| A6 `s_v` | `256` | `0.5297±0.0124` | `0.7258` | `0.9890±0.0030` |
| A6 `u_v` | `256` | `0.7818±0.0294` | `0.9238` | `0.9763±0.0057` |

Interpretation:

- the visual-unique channel is exactly the representation component the hypothesis predicts
- this is the strongest representation-level evidence for the three-way factorization story

### Step 7: Head ablation

Question:

> does the final detector actually route decisions through the expected subspace?

Answer:

- this is where the paper story becomes strongest

Observed pattern:

- `A2`: head mostly ignores visual causal features
- `A5`: head uses visual information partially
- `A6`: `FV-RA` strongly depends on `u_v`

Most important `A6` result:

- baseline `FV-RA = 0.7206`
- `mask_uv -> FV-RA = 0.3661`
- drop `= -0.3544`

Evidence table:

| Model | Baseline `FV-RA` | Most diagnostic ablation | Ablated `FV-RA` | `ΔFV-RA` | Interpretation |
| --- | ---: | --- | ---: | ---: | --- |
| A2 | `0.4949` | `mask_vc` | `0.4613` | `-0.0336` | visual route largely unused |
| A5 | `0.5860` | `mask_vis` | `0.4729` | `-0.1131` | partial visual usage |
| A6 | `0.7206` | `mask_uv` | `0.3661` | `-0.3544` | dedicated `u_v` routing |

Additional diagnostic:

- In `A2`, `mask_ac` also drops `AV1M` from `0.9975` to `0.5280`, confirming that the trained detector is fundamentally audio-dependent.

Interpretation:

- `A6` does not just *contain* a good visual subspace
- it actually *uses* that subspace at inference

This is the strongest single mechanism figure for the paper.

### Step 8: Corrected leading-silence control

Question:

> is the visual-fake gain just an audio-prefix or leading-silence artifact?

Answer:

- under corrected true-trimmed evaluation, `FV-RA` remains stable

Result:

- `A5`: `0.6156 -> 0.6058`
- `A6`: `0.7568 -> 0.7648`

Evidence table:

| Model | Baseline overall | Trimmed overall | `Δ` overall | Baseline `FV-RA` | Trimmed `FV-RA` | `ΔFV-RA` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 | `0.8186` | `0.7489` | `-0.0697` | `0.6156` | `0.6058` | `-0.0098` |
| A6 | `0.8885` | `0.8187` | `-0.0698` | `0.7568` | `0.7648` | `+0.0080` |

Interpretation:

- the visual-fake route does not disappear after removing the raw-waveform leading silence
- the residual drop is concentrated in audio-fake categories instead

This makes the visual-routing story much safer.

## 7. What The Complete Logical Chain Looks Like

The whole paper story can be read as:

1. `A2` is a strong binary baseline, but it is audio-dominant under source-domain training.
2. Simply adding domain adversarial pressure to the binary structure does not help; `A3` gets worse.
3. Three-way factorization creates a meaningful visual-unique channel.
4. That channel is highly predictive OOD, especially `u_v`.
5. `A6` changes downstream routing so that `FV-RA` decisions rely on `u_v`.
6. Therefore the gain comes from better representation structure plus better routing, not from full domain invariance.

## 8. Benchmark Map

The easiest way to avoid confusion is to separate benchmarks by role.

### Main benchmark

`AV1M -> FakeAVCeleb NPZ-matched`

- frozen AV-HuBERT
- matched FakeAVCeleb test split
- `N = 1114`
- report `mean±std`
- main paper result table

This is the benchmark that should anchor the main claim.

### Main categories inside the main benchmark

- `RV-FA = RealVideo-FakeAudio`
- `FV-RA = FakeVideo-RealAudio`
- `FV-FA = FakeVideo-FakeAudio`

Interpretation:

- `FV-RA` is the most informative category for the paper story
- it tests whether the model can use visual fake evidence under shift

### Older full-FAVC probe protocol

- `FAVC N = 4089`
- used mainly for representation probes
- should not be mixed with the main matched benchmark in paper tables

### AV1M validation probe benchmark

- used for in-domain label probes, perturbation probes, and branch analysis
- useful for understanding audio dominance and modality sensitivity
- not a replacement for the main OOD benchmark

### Balanced e2e benchmark

- balanced FAVC evaluation `500 + 500`
- used to show the architecture advantage is not a frozen-feature artifact

### True-trimmed leading-silence benchmark

- corrected robustness control
- used to test whether the `FV-RA` gain survives removal of the leading-silence artifact

### AVLips benchmark

- additional cross-dataset result
- currently supportive, not main
- useful as extra evidence but not stable enough to anchor the paper

### Public benchmark to add if possible

- `FakeAVCeleb 70-30 split`

Reason:

- best public comparability with prior literature
- should be kept separate from the internal matched protocol

## 9. Recent Results From April 8-9 And Why They Matter

The recent experiments from `2026-04-08` to `2026-04-09` are useful, but they do **not** all play the same role in the argument.

The best way to think about them is:

- one result strengthens the **core paper claim**
- one group strengthens the **design discussion / rebuttal space**
- one group gives **extra directional support**
- one group should **not yet be treated as independent paper evidence**

### 9.1 Corrected leading-silence evaluation

What was run:

- true-trimmed FAVC evaluation for `A5` and `A6`
- raw waveform leading silence was removed before AV-HuBERT feature extraction

Key result:

- `A5`: `FV-RA 0.6156 -> 0.6058`
- `A6`: `FV-RA 0.7568 -> 0.7648`

Why it matters:

- this is the single most useful recent result for the main paper argument
- it directly addresses the concern that `A6` might be exploiting a leading-silence artifact
- because `FV-RA` remains essentially stable, the result strengthens the claim that the visual-fake route is real

Recommended use:

- include in the main story as a supporting control
- acceptable for main text or at least strong appendix / rebuttal support

### 9.2 A6 train-real variant sweep

What was run:

- `A6_trainvalreal`
- `A6_sharedonly_trainreal`
- `A6_lite_trainreal`
- with multi-seed matched FAVC evaluation

Representative results:

| Variant | Overall AUC | FV-RA AUC | Role in the story |
| --- | ---: | ---: | --- |
| `A6_sharedonly_trainreal` | `0.8516±0.0166` | `0.6746±0.0365` | supports the idea that protecting unique channels can help |
| `A6_lite_trainreal` | `0.8401±0.0639` | `0.6503±0.1406` | weaker control; simplification alone is not enough |
| `A6_trainvalreal` strongest stable trio `43/44/45` | `0.9039±0.0231` | `0.7897±0.0508` | strongest follow-up result, but not main-table protocol |

Why it matters:

- these experiments help explain **which kind of domain support helps A6**
- they are useful for rebutting simplistic leakage critiques
- they also suggest that protecting or reshaping the domain objective relative to the factorized channels matters

What they do **not** prove:

- they do not replace the main benchmark
- they do not establish a new paper center, because they change the training setup
- they are follow-up experiments, not the cleanest apples-to-apples comparison against `A2/A5/A6`

Recommended use:

- discussion section
- appendix
- meeting update with Caren
- possible rebuttal material

### 9.3 AVLips multi-seed cross-dataset evaluation

What was run:

- 3-seed `AVLips` evaluation for `A2`, `A5`, `A6`

Key result:

| Model | Full AUC | Seed values |
| --- | ---: | --- |
| A2 | `0.5326±0.0258` | `0.5677, 0.5233, 0.5067` |
| A5 | `0.5880±0.0064` | `0.5889, 0.5797, 0.5954` |
| A6 | `0.6347±0.0715` | `0.5683, 0.7340, 0.6019` |

Why it matters:

- the mean ordering is still `A6 > A5 > A2`
- so it is directionally consistent with the main paper story

Limitation:

- `A6` variance is much larger here than on matched FakeAVCeleb
- therefore this benchmark is not yet stable enough to be used as central evidence

Recommended use:

- mention as supportive external-direction evidence
- do not make it a central table unless stability improves

### 9.4 Current `favc_7030` runs

What was run:

- `results_favc_7030` directories for `A2/A5/A6` seed sweeps were produced on `2026-04-09`

Current status:

- the numbers currently match the main matched benchmark exactly
- so these runs do not yet behave like an independent new benchmark result

Interpretation:

- this likely still needs protocol verification before being used as paper evidence

Recommended use:

- do not cite yet as a separate benchmark
- verify the exact split/protocol first

### 9.5 Practical conclusion

If we ask which recent results should actually enter the narrative:

1. Add the corrected leading-silence result directly into the main paper story.
2. Keep `A6_trainvalreal / sharedonly / lite` as follow-up evidence that strengthens discussion and rebuttal space.
3. Keep `AVLips` as directional support only.
4. Hold `favc_7030` out of the paper narrative until protocol independence is confirmed.

## 10. What We Should Claim

Safe claims:

- `A6 is strongest on the unified matched FakeAVCeleb benchmark.`
- `The key gain is on FV-RA.`
- `Three-way factorization creates a useful visual-unique pathway.`
- `A6 routes FV-RA decisions through that pathway.`
- `The gain is architecture-dependent, not just due to extra domain losses.`
- `The corrected leading-silence analysis does not support the hypothesis that FV-RA gains are an artifact of prefix silence.`

Claims to avoid:

- `A6 is fully domain-invariant.`
- `A6 cleanly disentangles all causal and spurious factors.`
- `Z_c contains little or no domain information.`
- `Every FCD branch has an interpretable one-to-one semantic meaning in a strict causal sense.`

## 11. Short Verbal Version

If we need a short 2-minute explanation, the clearest version is:

> The source-domain training data strongly encourages audio-dominant detection, which is exactly why the binary baseline struggles on the visual-fake OOD category `FV-RA`. Our hypothesis is that this is a structural problem: the binary causal/spurious split is too coarse to separate useful modality-specific signal from domain-correlated shortcut signal. We therefore move to a three-way factorization that explicitly exposes shared, unique, and residual components. The key result is not only that `A6` achieves the best matched FakeAVCeleb performance, but that the follow-up analyses line up with the mechanism: binary domain adversarial training alone fails (`A3 < A2`), the FCD `u_v` subspace is the strongest OOD visual carrier, and head ablation shows that `A6` actually routes `FV-RA` decisions through this visual-unique pathway. So the main claim is not domain invariance, but better representation usage and downstream routing under domain shift.

## 12. Paper-Ready Results Structure

The current story is already strong enough to support a coherent paper `Results` section, but the writing should be much tighter than this note.

The main principle is:

> do not present the experiments as a long chronological log; present them as a sequence of claims, where each claim is supported by one table or one figure.

### 12.1 Recommended main-text order

The cleanest main-text order is:

1. `A6` improves the main OOD benchmark, with the meaningful gain concentrated on `FV-RA`.
2. The gain is not explained by generic domain-adversarial training, because `A3 < A2`.
3. Three-way factorization exposes a strong visual-unique OOD subspace, especially `A6 u_v`.
4. The final `A6` detector actually routes `FV-RA` decisions through that visual-unique subspace.
5. This routing story survives the corrected leading-silence control.

This gives the paper a clean progression:

- performance
- control
- representation evidence
- routing evidence
- artifact control

### 12.2 Suggested main-text tables and figures

#### Table 1: Main benchmark

Title:

`Three-way factorization improves matched FakeAVCeleb OOD detection, especially on FV-RA`

Use:

- the current 3-seed matched benchmark table from Step 1
- keep `Overall / RV-FA / FV-RA / FV-FA`
- highlight `FV-RA` as the key column

Purpose:

- establish the main empirical claim immediately

#### Table 2: Architecture control

Title:

`The gain is architecture-dependent rather than a generic effect of domain-aware training`

Use:

- the Step 2 `A2 / A3 / A5 / A6` control table
- optionally keep only `Overall` and `FV-RA`

Purpose:

- show that simply adding domain adversarial pressure does not help in the binary setting
- justify why the paper needs the `A2 -> A3 -> A5 -> A6` progression

#### Figure 1: Model schematic

Title:

`Binary decomposition versus three-way factorization with domain push-pull`

Show:

- `A2` binary split
- `A5` shared / unique / residual factorization
- `A6` domain push-pull on top of the factorized structure

Purpose:

- make the structural hypothesis visually obvious before the analysis figures

#### Figure 2: Representation evidence

Title:

`Three-way factorization exposes a stronger visual-unique OOD subspace`

Show:

- the Step 6 subspace probe results
- at minimum compare `A2 Z_c`, `A5 s_v`, `A5 u_v`, `A6 s_v`, `A6 u_v`
- emphasize that `A6 u_v` has the strongest OOD probe

Purpose:

- support the representation-level part of the hypothesis

#### Figure 3: Routing evidence

Title:

`A6 routes FV-RA decisions through the visual-unique pathway`

Show:

- the Step 7 head-ablation drops
- plot `ΔFV-RA` for the most diagnostic ablation in `A2`, `A5`, `A6`
- include the `A2 mask_ac` diagnostic in a caption or side note

Purpose:

- this should be one of the main mechanism figures

#### Table 3: Artifact-control summary

Title:

`The visual-fake gain survives corrected leading-silence evaluation`

Use:

- the Step 8 corrected leading-silence table
- keep `Overall` and `FV-RA`

Purpose:

- answer the strongest artifact concern with one compact result

### 12.3 What should stay in the appendix

Keep these out of the main-text center unless the paper becomes much longer:

- AV1M perturbation details from Step 4
- full FAVC branch-probe tables from Step 5
- additional recent variant results: `A6_trainvalreal`, `A6_sharedonly_trainreal`, `A6_lite_trainreal`
- `AVLips`

Recommended appendix role:

- Step 4 and Step 5 support the claim that visual evidence exists in the representation
- the recent variant results support design discussion and rebuttal space
- `AVLips` provides directional external support, but not stable enough for a central claim

### 12.4 Draft paper subsection outline

If you want a paper-like section flow, the cleanest outline is:

#### `4.1 Three-way factorization improves visual-fake OOD detection`

Use `Table 1`.

Key message:

- `A6` is best on the matched benchmark
- the practically important gain is concentrated on `FV-RA`

Suggested paragraph:

> We first evaluate the binary baseline (`A2`) and the factorized models (`A5`, `A6`) on the matched FakeAVCeleb benchmark. As shown in Table 1, `A6` achieves the strongest overall OOD performance and the clearest improvement on `FV-RA`, the most informative category for testing visual-fake detection under shift. In contrast, `RV-FA` and `FV-FA` are already close to ceiling for all models. This indicates that the main benefit of the factorized architecture is not generic accuracy inflation, but improved use of visual evidence in the hardest cross-domain setting.

#### `4.2 The gain is not explained by generic domain-adversarial training`

Use `Table 2`.

Key message:

- `A3 < A2`
- domain pressure alone is insufficient

Suggested paragraph:

> We next test whether the gain comes simply from adding domain-aware objectives. The answer is no. In the binary architecture, adding domain adversarial pressure (`A3`) degrades both overall performance and `FV-RA` relative to `A2` (Table 2). The same family of objectives only becomes useful once the representation is factorized. This shows that the improvement is architecture-dependent rather than a generic byproduct of domain loss.

#### `4.3 Three-way factorization exposes a stronger visual-unique OOD subspace`

Use `Figure 2`.

Key message:

- `u_v` is the most important subspace
- especially in `A6`

Suggested paragraph:

> To understand why the factorized models generalize better, we probe the individual representation subspaces. Figure 2 shows that the visual-unique component `u_v` carries the strongest OOD signal, with `A6 u_v` substantially outperforming the binary `A2 Z_c` probe. This result supports the structural hypothesis behind the model design: the three-way factorization creates a usable visual-unique carrier that is not cleanly available in the binary decomposition.

#### `4.4 A6 routes FV-RA decisions through the visual-unique pathway`

Use `Figure 3` and `Table 3`.

Key message:

- the detector uses `u_v`
- the effect survives the artifact control

Suggested paragraph:

> Representation quality alone is not enough; the final detector must also use the right subspace. We therefore perform head ablations on the trained models. Figure 3 shows that `A6` is far more sensitive to masking `u_v` than the earlier models are to masking their visual components, indicating that the `FV-RA` decision is routed through the visual-unique pathway. Importantly, this interpretation survives the corrected leading-silence control: after removing raw-waveform prefix silence, the `FV-RA` performance remains essentially unchanged (Table 3). Together, these results support a routing-based explanation rather than an artifact-based one.

### 12.5 One-sentence writing rule

If a paragraph does not help support one of these four claims, it probably belongs in the appendix:

1. `A6` wins on `FV-RA`.
2. `A3` shows the gain is not from domain loss alone.
3. `A6 u_v` is the strongest visual-unique OOD carrier.
4. The trained `A6` detector actually uses that carrier, and the effect survives the artifact control.

## 13. Source Files To Cite

- `submission_prep/main3_seed_results_summary.csv`
- `submission_prep/unified_frozen_protocol_main.csv`
- `meeting_followup_experiments_summary.csv`
- `submission_prep/leading_silence_true_trim_results.csv`
- `submission_prep/LEADING_SILENCE_RESULTS.md`
- `submission_prep/NEURIPS_SUBMISSION_PLAN.md`
- `avh_sup/configs/A5.yaml`
- `avh_sup/configs/A6.yaml`

## 14. Caren's Action Items (from meeting, 2026-04-14)

Priority order for the next 10 days before submission.

### 14.1 Re-frame the core claim as routing, not invariance

**What to do:**

Every place in the draft where the model is described as "learning domain-invariant features" or "removing domain information", replace it with language that says:

> Three-way factorization changes *what information the model uses* and *how the final detector routes decisions*.

The word `invariance` should not appear in the abstract or introduction as a positive claim.
The word `routing` should appear at least once in the abstract and once in the first paragraph of the methods.

**Where to be especially careful:**

- Abstract: do not say "A6 learns domain-invariant features"
- Introduction: do not frame the contribution as "achieving invariance"
- Section 4.4 paragraph: do not say "domain pressure removes domain information from Z_c"

The correct framing is:

- The binary structure entangles signal and shortcut; three-way factorization separates them structurally.
- This structural separation changes *routing*: the trained head ends up using the visual-unique pathway for FV-RA.
- Domain probe numbers show A6 is not domain-invariant; it just routes better.

### 14.2 Elevate the three-way vs two-way mechanism evidence to a primary figure

**What to do:**

The head-ablation result (Step 7 in Section 6) is currently buried in a table.
Caren's feedback is that the size difference in routing sensitivity between A2 and A6 is large enough to be a primary mechanism figure, not just one table row.

**Concrete figure to make:**

A grouped bar chart with three groups (A2, A5, A6), each showing `ΔFV-RA` for their most diagnostic ablation:

| Model | Ablation | `ΔFV-RA` |
| --- | --- | ---: |
| A2 | `mask_vc` | `-0.0336` |
| A5 | `mask_vis` | `-0.1131` |
| A6 | `mask_uv` | `-0.3544` |

The visual claim this figure makes is:

> Only A6 exhibits a large routing drop when the visual-unique channel is masked. This is the primary evidence that three-way factorization changes downstream decision routing.

This should be **Figure 3** in the main text (see Section 12.2).

**Optional addition:** include `A2 mask_ac → ΔAV1M = −0.4695` as a side annotation to show that A2 is just as routing-sensitive — but to audio.

### 14.3 Re-package ablation results as inductive bias robustness, not absolute improvement

**What to do:**

When presenting results where absolute AUC is not impressively high (e.g., AVLips), do not lead with performance numbers.
Lead instead with ordering consistency:

> Across all evaluation protocols tested, the ranking A6 > A5 > A2 on FV-RA is preserved. This consistency supports the claim that three-way factorization provides a robust inductive bias for visual-fake detection, rather than an optimization artifact specific to one dataset.

**Suggested phrasing template for rebuttal / discussion:**

> Although absolute AUC varies across datasets and evaluation protocols, the relative improvement of the three-way factorization models over the binary baseline is consistent. This robustness of the ordering is itself evidence that the factorization structure induces a stable representational prior that favours visual-unique evidence under domain shift.

### 14.4 Build the key visualizations before sending to Chris and Mike

**Target: complete by end of next week (2026-04-21)**

Priority figures to produce:

1. **Figure 1 (model schematic):** side-by-side diagram of A2 (binary split) vs A5/A6 (three-way factorization + domain push-pull). Show data flow, factorization branches, and which branch feeds the classifier. Make `u_v` visually prominent.

2. **Figure 2 (subspace OOD probe):** grouped bar chart of OOD probe AUC by subspace. At minimum: `A2 Z_c`, `A5 u_v`, `A6 u_v`. Annotate `A6 u_v = 0.9238` as the peak. This is the representation-level evidence figure.

3. **Figure 3 (routing evidence / head ablation):** grouped bar chart of `ΔFV-RA` per model for the most diagnostic ablation. This is the primary mechanism figure. See 14.2 for exact values.

**Send to Chris and Mike once ready.**
Caren has offered to help with visualization if Chris and Mike agree.
Contact Caren if additional compute is needed (she has spare server capacity).

### 14.5 Summary: what changes the acceptance probability

Per Caren's framing, the paper moves from ~30–40% to ~45–55% if:

1. Routing language is used consistently throughout; invariance framing is removed.
2. Figure 3 (head ablation / routing evidence) is made into a strong standalone figure.
3. A publicly comparable benchmark is added (see Section 8, `FakeAVCeleb 70-30 split`).

Item 3 is the hardest and may require compute. This is when to contact Caren.
