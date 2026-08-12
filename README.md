# FPBench

FPBench evaluates foundation potentials (FPs) using application-oriented metrics across three
computational materials workflows:

1. **Force prediction** for atomistic simulations
2. **Phase stability and elemental ordering**
3. **Ion migration by NEB calculations**

Each component targets a specific computational task and evaluates FPs against metrics chosen
for their relevance to that task, rather than a single generic accuracy score.

**The current public release provides the Force Prediction component.**
**The Phase Stability and Elemental Ordering and Ion Migration by NEB components will be added later.**

## Components

| Component | Status | Documentation |
|---|---|---|
| Force Prediction | Available | [`Force_error/README.md`](Force_error/README.md) &middot; [website](https://kamirian.github.io/FPBench/force-error.html) |
| Phase Stability and Elemental Ordering | To be added | [`Phase_stability_ordering/README.md`](Phase_stability_ordering/README.md) |
| Ion Migration by NEB | To be added | [`Ion_migration_NEB/README.md`](Ion_migration_NEB/README.md) |

## Force Prediction

Evaluates FPs against DFT reference forces on MatPES-PBE, MatPES-r2SCAN, and OMat24 rattled-1000,
using metrics chosen for their relevance to structural relaxation, molecular dynamics, and NEB
calculations: highly accurate force predictions, joint force magnitude-angle accuracy,
large-force-error atoms, and force errors on far-from-equilibrium atoms.

See [`Force_error/README.md`](Force_error/README.md) for the full description, notation, and
quick start, and the [leaderboard](https://kamirian.github.io/FPBench/force-error.html) for
the current per-dataset results.

## Foundation Potentials Evaluated

Exact checkpoints and versions used in this study. Model size and training-dataset size are the
values used in this study's own code/records; official source links point to each architecture's
primary release page.

| FP | Model &amp; version | Model size | Training dataset | Approx. training-dataset size | Official source |
|---|---|---|---|---|---|
| MACE | &gt;=v0.3.10 (MACE-MPA-0, medium) | 9.06M | MPtrj + sAlex | ~3.5M | [MACE foundation models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) |
| CHGNet | v0.3.0 | 412.5K | MPtrj | ~1.58M | [CHGNet (GitHub)](https://github.com/CederGroupHub/chgnet) |
| M3GNet | MP-2021.2.8-PES | 288.2K | MP-2021.2.8 | ~176.6K | [MatGL](https://matgl.ai/) |
| UMA | s-1p1 | 146.5M | OC20 + ODAC23 + OMat24 + OMC25 + OMol25 | ~500M | [FAIR Chemistry](https://fair-chem.github.io/) |
| M3GNet-MatPES | v2025.1 | 664.2K | MatPES-PBE | ~435K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| TensorNet-MatPES | v2025.1 | 837.9K | MatPES-PBE | ~435K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| MACE-MatPES | &gt;=v0.3.10 | 9.06M | Fine-tuned on MatPES-PBE* | ~435K | No independent public checkpoint (fine-tuned for this study) |
| M3GNet-MatPES-r2SCAN | v2025.1 | 664.2K | MatPES-r2SCAN | ~388K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| TensorNet-MatPES-r2SCAN | v2025.1 | 837.9K | MatPES-r2SCAN | ~388K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| MACE-MatPES-r2SCAN | &gt;=v0.3.10 | 9.06M | Fine-tuned on MatPES-r2SCAN* | ~388K | No independent public checkpoint (fine-tuned for this study) |

\* Pre-trained on MACE-OMAT-0, then fine-tuned on the matched MatPES functional.

This is the current set of FPs evaluated. As additional FPs are tested, they will be added to this
table; existing benchmark results remain tied to their original model versions and are not
silently replaced.

---

## Evaluation Matrix

Which FPs were evaluated on which FPBench component/dataset. A checkmark means the FP was
evaluated there; the parenthetical states whether that dataset falls inside (`training`) or
outside (`OOD`, out-of-distribution) the FP's own training data, or whether the FP is a fine-tune
of an OOD base model onto that dataset (`fine-tuned`).

| FP | MatPES-PBE (Force Prediction) | OMat24 rattled-1000 (Force Prediction, SI) | Phase Stability/Ordering &amp; Ion Migration (NEB) |
|---|---|---|---|
| MACE | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| CHGNet | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| M3GNet | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| UMA | &#10003; (OOD) | &#10003; (training) | &#10003; (OOD) |
| M3GNet-MatPES | &#10003; (training) | &#10003; (OOD) | &#10003; (OOD) |
| TensorNet-MatPES | &#10003; (training) | &#10003; (OOD) | &#10003; (OOD) |
| MACE-MatPES | &#10003; (training) | &#10003; (fine-tuned)* | &#10003; (OOD) |

\* Pre-trained on MACE-OMAT-0, fine-tuned on MatPES-PBE.

The three r2SCAN-trained FPs (M3GNet-MatPES-r2SCAN, TensorNet-MatPES-r2SCAN, MACE-MatPES-r2SCAN)
are evaluated on the MatPES-r2SCAN dataset (training), in addition to the models listed above.

The "Phase Stability/Ordering & Ion Migration (NEB)" column reflects evaluation in the manuscript
only. **A checkmark in that column does not mean public code or results are available yet.** See
[Phase_stability_ordering/README.md](Phase_stability_ordering/README.md) and
[Ion_migration_NEB/README.md](Ion_migration_NEB/README.md), both still placeholders.

---

## Repository Structure

```text
FPBench/
├── README.md
├── LICENSE
├── Force_error/                     # Force Prediction component (available)
│   ├── README.md
│   ├── analysis/
│   ├── generation/
│   ├── scripts/
│   ├── data/
│   ├── examples/
│   └── requirements.txt
├── Phase_stability_ordering/        # to be added
│   └── README.md
├── Ion_migration_NEB/                # to be added
│   └── README.md
└── docs/                             # leaderboard website (GitHub Pages)
    ├── index.html
    ├── force-error.html
    ├── phase-stability-ordering.html
    └── ion-migration-neb.html
```

## Website

https://kamirian.github.io/FPBench/

## License

MIT. See [LICENSE](LICENSE).
