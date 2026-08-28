# FPBench

FPBench measures how well foundation potentials (FPs) perform on real computational materials
tasks, rather than on a single average accuracy score. It provides the reference datasets, the
evaluation code, and a public leaderboard for three workflows: force prediction, phase stability
and elemental ordering, and ion migration by NEB.

**Leaderboard: https://mogroupumd.github.io/FPBench/**

---

## How to use FPBench

There are three things people usually come here to do.

| I want to ... | Start here |
|---|---|
| **See how the tested FPs compare** | The [leaderboard](https://mogroupumd.github.io/FPBench/). No installation. |
| **Run FPBench on my own potential** | [Adding a potential](ADDING_A_POTENTIAL.md), then the component you care about. |
| **Apply the FPBench metrics to my own dataset** | The analysis functions in each component's `scripts/`. Each component README has a "Using FPBench" section describing this route. |

Every component follows the same shape. You produce a *standardized results file*, the validator
checks it, and the analysis functions turn it into the metric tables you see on the leaderboard.
Whether those results came from our reference dataset or from your own data does not change
anything downstream.

```text
        your FP + FPBench reference data
                      or
          your own DFT and FP results
                       |
              standardized results
                       |
              validation + analysis
                       |
                  metric tables
```

Each component documents its own version of this in detail:

- [Force prediction -- Using FPBench](Force_error/README.md#using-fpbench)
- [Phase stability and ordering -- Using FPBench](Phase_stability_ordering/README.md#using-fpbench)
- [Ion migration by NEB -- Using FPBench](Ion_migration_NEB/README.md#using-fpbench)

---

## Try it in five minutes

This runs on a real 5-structure slice of MatPES-PBE that ships with the repository. No large
download is needed.

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

That is the same standardization step every FPBench force metric is built on. To see the full
tables and figures, install `jupyterlab` and open an analysis notebook:

```bash
pip install jupyterlab
jupyter lab analysis/force_error_analysis_matpes_pbe.ipynb
```

---

## Benchmark your own potential

Three steps, the same for all three components.

1. **Register your FP.** Add one entry to the `POTENTIAL_REGISTRY` in the component's generator
   notebook, describing the environment to use and how to build an ASE calculator.
   See [Adding a potential](ADDING_A_POTENTIAL.md).
2. **Run the calculations.** The generator notebook writes job and submission scripts for your
   own cluster. Running the generation cells never submits anything; submission is always a
   separate, explicit step.
3. **Merge and analyse.** The generator merges the finished jobs into a standardized results
   file, and the analysis notebook produces the metric tables.

Requirements are per component; each has its own `requirements.txt`. We recommend a separate
virtual environment per FP family, since their PyTorch and ASE version requirements often
conflict.

---

## The three benchmarks

**[Force prediction](Force_error/README.md)** compares FP and DFT forces atom by atom on
MatPES-PBE, MatPES-r2SCAN, and OMat24 rattled-1000. Beyond average MAE and RMSE it reports the
fraction of highly accurate predictions, joint force magnitude-angle accuracy, large-force-error
atoms, and errors on far-from-equilibrium atoms.
[Results](https://mogroupumd.github.io/FPBench/force-error.html)

**[Phase stability and elemental ordering](Phase_stability_ordering/README.md)** tests whether an
FP preserves *relative* energies, on a chalcogenide phase-change-material dataset of 597 unique
hull candidates across 22 tie-line systems and 305 ordering groups. It reports ground-state
agreement, within-phase and global hull-minimum agreement, Top-1, Recall@k and Spearman ranking
metrics, and relaxation RMSD against the DFT-relaxed structure.
[Results](https://mogroupumd.github.io/FPBench/phase-stability-ordering.html)

**[Ion migration by NEB](Ion_migration_NEB/README.md)** runs the nudged elastic band method on
154 real Li- and Na-ion migration pathways across 109 unique structures. It reports non-converged
FP-NEB calculations, forward and backward barrier error, endpoint energy ranking and difference,
energy-profile shape agreement, endpoint relaxation error, and along-path force errors. Running
the full FP-NEB workflow alongside static FP evaluations on the DFT-NEB images separates
intrinsic potential-energy-surface error from error introduced by FP relaxation and pathway
optimization.
[Results](https://mogroupumd.github.io/FPBench/ion-migration-neb.html)

---

## Repository structure

```text
FPBench/
├── README.md
├── ADDING_A_POTENTIAL.md             # how to register and run your own FP
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

\* Pre-trained on MACE-OMAT-0, then fine-tuned on the matched MatPES functional.

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

\* Pre-trained on MACE-OMAT-0, fine-tuned on MatPES-PBE.

The three r2SCAN-trained FPs are evaluated on MatPES-r2SCAN (training), in addition to the
models above.

---

## License

MIT. See [LICENSE](LICENSE).
