# Force Prediction

The Force Prediction component of [FPBench](../README.md). Evaluates foundation potentials (FPs)
against DFT reference forces on three datasets: **MatPES-PBE**, **MatPES-r2SCAN**, and
**OMat24 rattled-1000**.

Results website: **https://kamirian.github.io/FPBench/force-error.html**

---

## Application-Oriented Force Metrics

FPBench evaluates FP force predictions using four metrics chosen for their direct relevance to
atomistic simulation workflows.

**Highly accurate force predictions (small-force-error atoms).** Strict force convergence
criteria of 0.001-0.01 eV/Å are often used to relax structures for subsequent calculations such
as defect energies, electronic structure, and NEB. Force errors comparable to these convergence
thresholds can drive the relaxation toward a different minimum or cause it to stop before the
correct DFT equilibrium structure is reached.

**Joint force magnitude-angle accuracy.** Joint force magnitude-angle accuracy measures the
fraction of atoms that simultaneously satisfy the |Δ|F|| and Δθ thresholds. This analysis shows
whether both force magnitude and force angle are predicted accurately for the same atom and
reveals the difficulty of achieving high accuracy in both quantities simultaneously.

**Large-force-error atoms.** Large force errors are highly undesired in atomistic simulations
using FPs because even a single occurrence can lead to significant problems, such as erroneous
atomic dynamics in MD simulations or failure of convergence in NEB calculations. The fraction of
large-force-error atoms is therefore a key FP error metric.

**Force errors on far-from-equilibrium (FE) atoms.** Many computational tasks involve
far-from-equilibrium atoms, including migrating ions, defect and disordered structures, and large
thermal displacements in high-temperature MD simulations. Force errors on FE atoms should be
evaluated separately; otherwise, they can be diluted by the dominant near-equilibrium population.

Together, these metrics evaluate force-prediction behavior directly relevant to structural
relaxation, MD, NEB, and other atomistic simulations.

---

## Notation

```text
Δ|F| = |F_FP| − |F_DFT|
|Δ|F||
Δθ
e_vec = ||F_FP − F_DFT||
r_F = |Δ|F|| / |F_DFT|
```

`Δ|F| MAE/RMSE` is used for MAE/RMSE analyses. `|Δ|F||` is used for CDFs and threshold/fraction
analyses.

**Zero-force and evaluated-population wording.** Atoms with zero FP or DFT force were excluded
from all metrics, since the force angle is undefined in these cases. Except for the all-atom
MAE/RMSE analysis, all force metrics were evaluated only for atoms with |F_DFT| > 0.01 eV/Å,
because for near-zero DFT forces, small absolute differences in the force components can produce
large changes in the calculated force angle.

```text
evaluated: |F_DFT| > 0.01 eV/Å
non-FE:    0.01 < |F_DFT| <= 1 eV/Å
FE:        |F_DFT| > 1 eV/Å
```

---

## Datasets

| Dataset | DFT functional | FPs evaluated |
|---|---|---|
| MatPES-PBE | PBE | 7 |
| MatPES-r2SCAN | r2SCAN | 3 (the r2SCAN-trained FPs only) |
| OMat24 rattled-1000 | PBE | 7 (same FPs as MatPES-PBE) |

MatPES-r2SCAN contains only the three r2SCAN-trained FPs; no PBE-trained model is mixed in.
OMat24 rattled-1000 is the PBE-labeled OMat24 rattled-1000 validation dataset, evaluated with the
same seven PBE-trained FPs used on MatPES-PBE.

Dataset-level atom counts and FE fractions are not reported here. The currently-committed
standardized files predate the generators' dataset-wide `reference_population` field (see
Section 7 of the generator notebooks), so a model-independent count has not been calculated.
Per-FP evaluated-atom counts differ slightly across FPs, since each FP excludes its own zero-force
atoms, and are not a substitute for a dataset-level count.

---

## Standardized Output Schema

Each dataset has one standardized results file:

```text
matpes_pbe_force_results_standardized.json
matpes_r2scan_force_results_standardized.json
omat24_rattled_1000_force_results_standardized.json
```

```json
{
  "schema_version": "1.0",
  "dataset_name": "...",
  "units": {"force": "eV/Å", "angle": "degree"},
  "generation_metadata": {...},
  "reference_population": {...},
  "models": {
    "<fp_name>": {
      "dft_force_magnitude":   [...],
      "fp_force_magnitude":    [...],
      "force_magnitude_error": [...],
      "force_angle_error":     [...],
      "force_vector_error":    [...],
      "structure_id":          [...],
      "atom_index":            [...]
    }
  }
}
```

`reference_population` is built once from the raw, unfiltered DFT-only data. It is independent of
any one FP's zero-force exclusions and is what a dataset-level force distribution or FE fraction
should be computed from, not a model-specific filtered array. Each entry under `models` contains
only that FP's own metric-valid atoms, plus `structure_id`/`atom_index` provenance.

Raw, per-structure Cartesian forces (both DFT and FP) are not in the standardized file. They
remain in the generator's per-chunk `results_*.json` checkpoints, together with every atom
(including the ones excluded from the standardized file's metric arrays) and a `null` angle
wherever it is undefined.

---

## Repository Structure

```text
Force_error/
├── README.md
├── analysis/
│   ├── force_error_analysis_matpes_pbe.ipynb
│   ├── force_error_analysis_matpes_r2scan.ipynb
│   └── force_error_analysis_omat24_rattled_1000.ipynb
├── generation/
│   ├── matpes_PBE_run_generator.ipynb
│   ├── matpes_r2scan_run_generator.ipynb
│   └── matpes_run_generator_omat24_rattled1000.ipynb
├── scripts/
│   ├── force_results.py
│   ├── force_error_metrics.py
│   ├── fp_cdf_density_plots.py
│   └── heatmap_table.py
├── examples/
│   └── mace_matpes_cartesian_force_example.json
├── data/
│   └── README.md
└── requirements.txt
```

---

## Using FPBench with Another Dataset

1. **Analysis only, from paired Cartesian forces.** Call
   `build_force_results(dft_forces, fp_forces, structure_ids=None)` from `scripts/force_results.py`
   directly, see the worked example in `analysis/force_error_analysis_matpes_pbe.ipynb`.
2. **Full dataset generation.** Use one of the three notebooks in `generation/` as a template. Each
   generator notebook writes SLURM job and submission scripts; it does not automatically submit
   jobs merely by running the generation cells. Submission still requires an explicit step on your
   cluster.
3. **`structure_id`/`atom_index` are preserved throughout** the generation workflow, from the raw
   chunk through the per-structure checkpoint into the final standardized file, so any atom can be
   traced back to its source structure and position within it.
4. **Configuration.** Every dataset/cluster/checkpoint path is a plain, clearly marked variable
   near the top of its cell. Cells raise a clear error if run before the placeholders are edited.
   We recommend a separate virtual environment per FP family, since their dependency requirements
   can conflict, and the notebooks never install packages automatically.

---

## Python Modules (`scripts/`)

- **`force_results.py`**: `build_force_results(...)`, the canonical standardization function.
  The generator notebooks embed a checksummed, byte-identical reimplementation of its
  per-structure math so cluster jobs do not need this module installed.
- **`force_error_metrics.py`**: fraction tables, MAE/RMSE, regime panels, heatmaps, and
  `get_bad_atom_indices` querying.
- **`fp_cdf_density_plots.py`**: CDF computation and 2-D density plotting.
- **`heatmap_table.py`**: low-level heatmap drawing primitives.

---

## Quick Start

```bash
git clone https://github.com/kamirian/FPBench.git
cd FPBench/Force_error
pip install -r requirements.txt
pip install jupyterlab
jupyter lab analysis/force_error_analysis_matpes_pbe.ipynb
```

- The small Cartesian-force example (`examples/mace_matpes_cartesian_force_example.json`) is
  available immediately and demonstrates `build_force_results(...)` without any additional
  download.
- Reproducing the full analysis requires the large standardized JSON files (see
  [`data/README.md`](data/README.md)). These are not committed to Git. Their public deposition
  links will be added here when available.
- The `generation/` notebooks can reproduce the standardized files given the raw datasets, FP
  checkpoints, separate per-FP environments, and cluster resources. They are not required to
  explore the analysis code.

Not every result on the website is reproducible immediately from a clone of this repository;
full reproduction depends on the standardized files described above.

---

## Citation

The manuscript this work supports is not yet publicly available. Final citation information will
be added here once it is. In the meantime:

```bibtex
@misc{amirian2026fpbench,
  author = {Kiyan Amirian and Yifei Mo},
  title  = {FPBench: Foundation Potential Force-Error Analysis},
  year   = {2026},
  url    = {https://github.com/kamirian/FPBench}
}
```

---

## License

MIT. See the [repository-level LICENSE](../LICENSE).
