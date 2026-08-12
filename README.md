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
quick start, and the [results website](https://kamirian.github.io/FPBench/force-error.html) for
the current per-dataset results.

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
└── docs/                             # results website (GitHub Pages)
    ├── index.html
    ├── force-error.html
    ├── phase-stability-ordering.html
    └── ion-migration-neb.html
```

## Website

https://kamirian.github.io/FPBench/

## License

MIT. See [LICENSE](LICENSE).
