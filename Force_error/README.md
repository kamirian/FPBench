# Force Prediction

The Force Prediction component of [FPBench](../README.md). Evaluates foundation potentials (FPs)
against DFT reference forces on three datasets: **MatPES-PBE**, **MatPES-r2SCAN**, and
**OMat24 rattled-1000**.

Leaderboard: **https://kamirian.github.io/FPBench/force-error.html**

---

## Application-Oriented Force Metrics

FPBench evaluates FP force predictions using four metrics chosen for their direct relevance to
atomistic simulation workflows.

**Highly accurate force predictions.** The fraction of atoms with a very small force-magnitude
error is directly relevant to applications that require tight force convergence, such as
structural relaxation, defect calculations, and transition-state searches, where typical
convergence criteria are on the order of 0.001-0.01 eV/Å.

**Joint force magnitude-angle accuracy.** This metric measures whether an FP reproduces both the
force magnitude and the force direction for the same atom. Separate magnitude and angle
statistics, taken on their own, do not establish this joint agreement.

**Large-force-error atoms.** A small population of atoms with large force errors can still affect
structural relaxation, transition-region calculations, and molecular-dynamics trajectories, even
when the average error looks acceptable.

**Force errors on far-from-equilibrium (FE) atoms.** Forces on high-DFT-force configurations are
important for assessing FP behavior away from equilibrium, including migrating ions, defect and
disordered structures, and large thermal displacements.

These metrics are complementary and should be interpreted together rather than combined into a
single overall ranking.

---

## Notation

The force error was decomposed into the force-magnitude error, `Δ|F| = |F_FP| − |F_DFT|`, and the
force-angle error, `Δθ`, defined as the angle between the FP and DFT force vectors. We also
computed the force-vector error, `e_vec = ||F_FP − F_DFT||`, and the relative force-magnitude
error for FE atoms, `r_F = |Δ|F|| / |F_DFT|`.

`Δ|F| MAE/RMSE` is used for the MAE/RMSE metric name. `|Δ|F||` is used for CDFs, thresholds, and
fraction metrics.

Atoms with zero FP or DFT force were excluded from all metrics, since the force angle is
undefined in these cases. Except for the all-atom MAE/RMSE analysis, all force metrics were
evaluated only for atoms with `|F_DFT| > 0.01` eV/Å, because for near-zero DFT forces, small
absolute differences in the force components can produce large changes in the calculated force
angle. FE atoms correspond to `|F_DFT| > 1` eV/Å.

---

## Datasets

| Dataset | DFT functional | FPs evaluated |
|---|---|---|
| MatPES-PBE | PBE | 7 |
| MatPES-r2SCAN | r2SCAN | 3 |
| OMat24 rattled-1000 | PBE | 7 |

The MatPES-r2SCAN analysis includes the three r2SCAN-trained FPs. The OMat24 rattled-1000 analysis
evaluates the same seven FPs used for MatPES-PBE.

Dataset-level atom counts and FE fractions are not reported here until they have been calculated
from the generators' dataset-wide `reference_population` field (see Section 7 of the generator
notebooks).

---

Exact checkpoints/versions, model sizes, and the full evaluation matrix (which FPs were run on
which dataset, across all FPBench components) are documented once at the
[FPBench home page](../README.md#foundation-potentials-evaluated) rather than duplicated here.

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

The three analysis notebooks (`analysis/force_error_analysis_matpes_pbe.ipynb`,
`analysis/force_error_analysis_matpes_r2scan.ipynb`,
`analysis/force_error_analysis_omat24_rattled_1000.ipynb`) and the three generator notebooks
(`generation/matpes_PBE_run_generator.ipynb`, `generation/matpes_r2scan_run_generator.ipynb`,
`generation/matpes_run_generator_omat24_rattled1000.ipynb`) are the same notebooks used to produce
the results reported in the FPBench manuscript.

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
5. **Adding a new FP to this benchmark.** Once you have run the steps above against MatPES-PBE,
   MatPES-r2SCAN, or OMat24 rattled-1000 and computed the metrics with the corresponding analysis
   notebook, open a [GitHub issue](https://github.com/kamirian/FPBench/issues) with the FP's name,
   architecture, training data, checkpoint/version, and computed metrics.

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
