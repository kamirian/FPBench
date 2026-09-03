# FPBench

FPBench is an application-oriented benchmark for foundation potentials (FPs). It evaluates FPs on
representative computational materials tasks rather than on average energy and force errors alone,
and provides the reference datasets, the evaluation code, and a public leaderboard for three
components: force prediction, phase stability and elemental ordering, and ion migration by NEB.

**Leaderboard: https://mogroupumd.github.io/FPBench/**

---

## Using FPBench

FPBench supports three modes of use.

- **Reported results.** The [leaderboard](https://mogroupumd.github.io/FPBench/) gives the
  current metrics for every evaluated FP and requires no installation.
- **Evaluation of an additional potential.** The FP is registered in the relevant generator
  notebook and run against the provided reference datasets. See
  [Adding a potential](ADDING_A_POTENTIAL.md).
- **Application to an independent dataset.** The analysis functions in each component's
  `scripts/` directory accept standardized inputs directly.

All three components share a common workflow. Calculations are written to a standardized results
file, validated, and passed to the analysis functions that produce the reported metric tables. The
provided reference data and user-supplied data enter this workflow at the same point.

```text
        FP evaluated on the FPBench reference data
                          or
            user DFT and FP results
                          |
                standardized results
                          |
               validation and analysis
                          |
                    metric tables
```

Each component documents its own workflow in detail.

- [Force prediction](Force_error/README.md#using-fpbench)
- [Phase stability and elemental ordering](Phase_stability_ordering/README.md#using-fpbench)
- [Ion migration by NEB](Ion_migration_NEB/README.md#using-fpbench)

---

## Quick start

The following example uses a five-structure subset of MatPES-PBE included in the repository and
requires no additional download.

```bash
git clone https://github.com/mogroupumd/FPBench.git
cd FPBench/Force_error
pip install -r requirements.txt
```

```python
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from force_results import build_force_results

with open("examples/mace_matpes_cartesian_force_example.json") as f:
    example = json.load(f)

force_results = build_force_results(
    example["dft_forces"],
    {"mace": example["fp_forces"]},
    structure_ids=example["structure_ids"],
)

print("Models:", list(force_results))
print("Fields:", list(force_results["mace"]))
```

This is the standardization step underlying every force metric reported by FPBench. The complete
tables and figures are produced by the analysis notebooks.

```bash
pip install jupyterlab
jupyter lab analysis/force_error_analysis_matpes_pbe.ipynb
```

---

## Evaluating a new potential

The procedure is the same for all three components.

1. **Register the potential.** Add one entry to the `POTENTIAL_REGISTRY` in the component's
   generator notebook, specifying the environment to use and the code that constructs an ASE
   calculator. See [Adding a potential](ADDING_A_POTENTIAL.md).
2. **Run the calculations.** The generator notebook writes job and submission scripts for the
   target cluster. Running the generation cells does not submit any job; submission is a separate
   step.
3. **Merge and analyse.** The generator merges the completed jobs into a standardized results
   file, which the analysis notebook converts into the metric tables.

Requirements are specified per component in the corresponding `requirements.txt`. A separate
virtual environment per FP family is recommended, since their PyTorch and ASE version
requirements frequently conflict.

---

## Benchmark components

**[Force prediction](Force_error/README.md)** compares FP and DFT forces atom by atom on
MatPES-PBE, MatPES-r2SCAN, and OMat24 rattled-1000. In addition to average MAE and RMSE, it
reports the fraction of highly accurate force predictions, joint force magnitude-angle accuracy,
large-force-error atoms, and errors on far-from-equilibrium atoms.
[Results](https://mogroupumd.github.io/FPBench/force-error.html)

**[Phase stability and elemental ordering](Phase_stability_ordering/README.md)** evaluates whether
an FP reproduces the relative energies of competing phases, compositions, and elemental orderings,
using a chalcogenide phase-change-material dataset of 597 unique hull structures across 22
tie-line systems and 305 ordering groups. It reports ground-state agreement, within-phase and
global hull-minimum agreement, Top-1 accuracy, Recall@k, Spearman rank correlation, and
relaxation RMSD against the DFT-relaxed structure.
[Results](https://mogroupumd.github.io/FPBench/phase-stability-ordering.html)

**[Ion migration by NEB](Ion_migration_NEB/README.md)** applies the nudged elastic band method to
154 Li- and Na-ion migration pathways spanning 106 structurally distinct materials. It reports non-converged
FP-NEB calculations, forward and backward barrier errors, endpoint energy ranking and
energy-difference errors, energy-profile shape agreement, endpoint relaxation error, and
along-path force errors. Comparing the full FP-NEB workflow with static FP evaluations on the
DFT-NEB images separates intrinsic potential-energy-surface error from error introduced by FP
endpoint relaxation and pathway optimization.
[Results](https://mogroupumd.github.io/FPBench/ion-migration-neb.html)

---

## Repository structure

```text
FPBench/
├── README.md
├── ADDING_A_POTENTIAL.md             # how to register and run an additional FP
├── LICENSE
├── Force_error/                      # Force prediction
│   ├── README.md
│   ├── analysis/                     # analysis notebooks
│   ├── generation/                   # job generators (POTENTIAL_REGISTRY lives here)
│   ├── scripts/                      # importable metric and standardization functions
│   ├── data/                         # data documentation and download pointers
│   ├── examples/                     # small runnable examples, committed
│   └── requirements.txt
├── Phase_stability_ordering/         # same layout
├── Ion_migration_NEB/                # same layout, plus tests/
└── docs/                             # leaderboard website (GitHub Pages)
```

Large standardized data files are not committed to Git. Each component's `data/README.md` says
where to obtain them; the `examples/` slices need no download.

---

## Foundation potentials evaluated

These are the FPs currently on the leaderboard, with the exact checkpoints used. Model and
training-set sizes are the values recorded by this study's own code. As more FPs are tested they
are added here; existing results stay tied to the model version they were produced with and are
not silently replaced.

| FP | Model &amp; version | Model size | Training dataset | Approx. training-set size | Source |
|---|---|---|---|---|---|
| MACE | &gt;=v0.3.10 (MACE-MPA-0, medium) | 9.06M | MPtrj + sAlex | ~3.5M | [MACE foundation models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) |
| CHGNet | v0.3.0 | 412.5K | MPtrj | ~1.58M | [CHGNet](https://github.com/CederGroupHub/chgnet) |
| M3GNet | MP-2021.2.8-PES | 288.2K | MP-2021.2.8 | ~176.6K | [MatGL](https://matgl.ai/) |
| UMA | s-1p1 | 146.5M | OC20 + ODAC23 + OMat24 + OMC25 + OMol25 | ~500M | [FAIR Chemistry](https://fair-chem.github.io/) |
| M3GNet-MatPES | v2025.1 | 664.2K | MatPES-PBE | ~435K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| TensorNet-MatPES | v2025.1 | 837.9K | MatPES-PBE | ~435K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| MACE-MatPES | &gt;=v0.3.10 | 9.06M | Fine-tuned on MatPES-PBE* | ~435K | [MACE](https://github.com/acesuit/mace) |
| M3GNet-MatPES-r2SCAN | v2025.1 | 664.2K | MatPES-r2SCAN | ~388K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| TensorNet-MatPES-r2SCAN | v2025.1 | 837.9K | MatPES-r2SCAN | ~388K | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| MACE-MatPES-r2SCAN | &gt;=v0.3.10 | 9.06M | Fine-tuned on MatPES-r2SCAN* | ~388K | [MACE](https://github.com/acesuit/mace) |
| Orb | orb-v3-conservative-inf-omat-20250404 | 25.5M | OMat24, AIMD subset only | ~55M&Dagger; | [Rhodes et al. 2025](https://arxiv.org/abs/2504.06231) |
| SevenNet | 7net-mf-ompa (modal `mpa`) | 25.7M | MPtrj + sAlex + OMat24, multi-fidelity; `mpa` selects the MPtrj + sAlex task | not reported for this checkpoint | [SevenNet pretrained models](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html) &middot; [Kim et al. 2025](https://doi.org/10.1021/jacs.4c14455) |
| MatterSim | MatterSim-v1.0.0-5M | 4.55M | Nonpublic MatterSim active-learning dataset, GGA-PBE(+U)&dagger; | 6M | [Model card](https://github.com/microsoft/mattersim/blob/main/MODEL_CARD.md) &middot; [Yang et al. 2024](https://arxiv.org/abs/2405.04967) |

\* Pre-trained on MACE-OMAT-0, then fine-tuned on the matched MatPES functional.

&dagger; The MatterSim model card reports "Training Data Size: 6M", "Model Parameters: 4.5M", and
training on "a specific variant of Density Functional Theory (PBE)". The GGA-PBE(+U) labelling,
with Hubbard U applied to selected materials following Materials Project settings, is described in
Yang et al. rather than on the model card. The dataset itself is not released, so the OOD label
reflects the absence of documented training exposure rather than a verified composition.

&Dagger; Rhodes et al. state that "all orb-v3-*-omat models are only trained on the AIMD subset of
OMat24", and that the OMat24 dataset "contains ~55 million AIMD-sampled structures".

Orb, SevenNet and MatterSim are additional potentials evaluated after submission of the
manuscript, which reports the ten FPs above them. Model sizes for these three are parameter counts measured from the loaded checkpoints, not
figures quoted from the papers.

### Evaluation matrix

Which FPs were evaluated on which dataset. The parenthetical says whether that dataset is inside
the FP's own training data (`training`), outside it (`OOD`), or a fine-tune of an OOD base model
onto it (`fine-tuned`).

| FP | MatPES-PBE (force) | OMat24 rattled-1000 (force, SI) | Phase stability / ordering and NEB |
|---|---|---|---|
| MACE | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| CHGNet | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| M3GNet | &#10003; (OOD) | &#10003; (OOD) | &#10003; (OOD) |
| UMA | &#10003; (OOD) | &#10003; (training) | &#10003; (OOD) |
| M3GNet-MatPES | &#10003; (training) | &#10003; (OOD) | &#10003; (OOD) |
| TensorNet-MatPES | &#10003; (training) | &#10003; (OOD) | &#10003; (OOD) |
| MACE-MatPES | &#10003; (training) | &#10003; (fine-tuned)* | &#10003; (OOD) |
| Orb | &#10003; (OOD) | not evaluated | not evaluated |
| SevenNet | &#10003; (OOD) | not evaluated | not evaluated |
| MatterSim | &#10003; (OOD)&dagger; | not evaluated | not evaluated |

\* Pre-trained on MACE-OMAT-0, fine-tuned on MatPES-PBE.

The three r2SCAN-trained FPs are evaluated on MatPES-r2SCAN (training), in addition to the
models above.

---

## License

MIT. See [LICENSE](LICENSE).
