# GranulationCNN

**A Granulation-Based Temporal Convolutional Network for Student Engagement Detection in E-Learning Environments**

Research internship project at the **Indian Institute of Management Ranchi**, under the supervision of **Prof. Sobhan Sarkar** (Department of Information Systems and Analytics).

---

## Overview

GranulationCNN is a lightweight, webcam-based engagement estimator for online learning. It starts from a simple observation: averaging a ten-second video clip into a single embedding throws away information that matters. Within one short window a student can look at the screen, glance away, and refocus — a mean over those frames smooths that variation into invisibility.

Instead of pooling all frame embeddings into one vector, the **Temporal Granulation Module (TGM)** randomly partitions them into several groups ("granules"), computes a mean prototype per granule, and averages the per-granule logits into a single prediction. The random partition changes every forward pass, which acts as an implicit ensemble over many temporal views of the same clip; a compactness term keeps each granule geometrically coherent.

The model is framed as a **decision-support signal for instructors** — surfacing trends and reducing monitoring burden — not an autonomous monitor making high-stakes calls about individual learners.

## Key Idea: The Temporal Granulation Module

1. Uniformly sample `T = 8` frames per clip and encode each with a shared **ResNet-18** into a 512-d embedding.
2. Stochastically assign each frame to one of `G = 4` granules.
3. Compute a mean prototype per non-empty granule.
4. Pass each prototype through a shared linear classifier and average the granule-level logits.

When `G = 1`, the TGM reduces exactly to standard temporal average pooling — so granulation is a strict generalization of pooling, not an alternative to it. Stochastic granulation adds training-time diversity **without a single extra parameter**.

## Dataset

- **DAiSEE** — a naturalistic benchmark for affective-state recognition in e-learning.
- Evaluation is scoped to the **C2 (moderate)** and **C3 (high)** engagement levels. C0 and C1 together cover under 5% of the test split — too few for standard cross-entropy to learn reliably. Severe disengagement (C0/C1) is treated as future work requiring purpose-built data collection.
- Binary evaluation scope: 1,553 clips (849 C2, 704 C3).

## Results

Three-variant ablation (one component changed per row), mean ± std across 3 seeds:

| Variant | Backbone | Loss | Accuracy | Weighted F1 | Cohen's κ |
|---|---|---|---|---|---|
| A | Random | CE | 0.511 ± 0.012 | 0.391 ± 0.026 | −0.001 ± 0.011 |
| B | ImageNet | Focal | 0.285 ± 0.056 | 0.332 ± 0.042 | 0.010 ± 0.008 |
| **C** | **ImageNet** | **CE** | **0.530 ± 0.004** | **0.418 ± 0.007** | **0.038 ± 0.007** |

**Headline (Variant C, representative seed 456):** 56.15% accuracy, 0.459 weighted F1, Cohen's κ = 0.0468 on the binary C2/C3 subset.

**Statistical validation vs. the C2-majority baseline:** McNemar χ² = 4.36, p = 0.037; both bootstrap confidence intervals (ΔAccuracy, ΔF1) stay above zero.

### What the ablation shows

- **Random initialization collapses.** Variant A is unreliable across seeds (κ from −0.017 to +0.016).
- **Pretraining is the decisive change.** It does not just raise the mean — it removes the instability (±0.004 accuracy across seeds).
- **Focal loss recovers minority classes** (Variant B reaches C3 recall 0.334) at the cost of aggregate accuracy, showing the architecture is not structurally biased toward majority-class prediction.
- **Granulation beats average pooling by 3.59 pp** on validation accuracy (`G = 4` vs. `G = 1`), the largest single gain in the hyperparameter sweep.

## Configuration

| Setting | Value |
|---|---|
| Frames per clip (`T`) | 8 |
| Granules (`G`) | 4 |
| Compactness weight (`λ`) | 0.3 |
| Backbone | ResNet-18 (IMAGENET1K_V1) |
| Optimizer | Adam, lr = 3e-4 |
| Batch size | 16 |
| Epochs | 5 |
| Hardware | Kaggle T4 GPU (AMP) |

## Why It Fits a Real Deployment

- **Light** — ResNet-18 plus a linear head can process many concurrent learner streams where large video transformers cannot.
- **Inspectable** — per-granule logits form a short within-clip evidence timeline an instructor can interrogate.
- **Uncertainty for free** — granule disagreement at inference is a natural low-confidence indicator, with no extra parameters or forward passes.

## Limitations

κ ≈ 0.04 is a modest number and is **not** a basis for autonomous decisions about individual learners. C3 recall under standard CE needs improvement, and C0/C1 engagement is a separate problem requiring different data and possibly different sensing modalities.

## Status

First-author manuscript prepared from this work. Code and experiment logs are available in the accompanying Kaggle notebook.

## Acknowledgements

Carried out as a research internship at IIM Ranchi under **Prof. Sobhan Sarkar**, with access to Kaggle GPU resources and the DAiSEE dataset.
