# FPBench: Foundation Potential Force-Error Analysis

A force-error analysis framework for foundation potentials (FPs — general-purpose machine-learning
interatomic potentials), evaluated against DFT reference forces on three datasets:
**MatPES-PBE**, **MatPES-r2SCAN**, and **OMat24 rattled-1000**.

**The central argument:** total average error (MAE/RMSE) alone is an incomplete benchmark. It is
diluted by the large fraction of near-equilibrium atoms and hides failure modes — misdirected forces,
rare but severe magnitude errors, poor performance on far-from-equilibrium configurations — that
matter for real atomistic simulations.

---

## Motivation

FP evaluations often report a single number: MAE or RMSE of the force-magnitude error across all
atoms. A model can have a low average error while still:

- Predicting forces with systematically wrong *directions* (large force-angle error Δθ), even when
  the magnitude looks fine
- Producing rare but severe force-magnitude errors that never show up in an average
- Doing well near equilibrium while failing on far-from-equilibrium (FE) configurations, which are
  under-represented in most training/evaluation sets but matter most for NEB, relaxations, and
  high-temperature MD

FPBench decomposes force accuracy across regime, magnitude, and direction instead of collapsing it
into one number.

---

## Notation

| Symbol | Definition |
|---|---|
| `Δ\|F\| = \|F_FP\| − \|F_DFT\|` | **Signed** force-magnitude error |
| `\|Δ\|F\|\|` | **Absolute** force-magnitude error |
| `Δθ` | Force-angle error (degrees) between `F_FP` and `F_DFT` |
| `e_vec = ‖F_FP − F_DFT‖` | Force-**vector** error (Euclidean norm of the vector difference) |
| `r_F = \|Δ\|F\|\| / \|F_DFT\|` | Relative force-magnitude error |

**Zero-force exclusion.** An atom's force angle is undefined if either `F_DFT` or `F_FP` is ~zero
for that atom. Such atoms are excluded from every flattened per-atom *metric* array, independently
per FP (a zero FP-predicted force is model-specific, so different FPs can retain slightly different
atom populations at this step — that is expected, not an inconsistency). The raw per-structure
checkpoint written by the generator notebooks (`results_*.json`) still retains **every** atom
regardless, with the undefined angle stored as `null`. The all-atom force-magnitude MAE/RMSE sweep
in the analysis notebooks is the one place that deliberately reports on the full population,
including near-zero-force atoms, as the explicit exception to the usual cutoff below.

**Evaluated population.**

```text
evaluated: |F_DFT| > 0.01 eV/Å      (FDFT_MIN cutoff; excludes near-zero DFT forces)
non-FE:    0.01 < |F_DFT| ≤ 1 eV/Å
FE:        |F_DFT| > 1 eV/Å
```

`FDFT_MIN` is applied only in the **analysis** notebooks, never during generation — the generator
notebooks and their raw per-chunk checkpoints keep the full atom population.

---

## Datasets

| Dataset | Functional | FPs evaluated | Notes |
|---|---|---|---|
| **MatPES-PBE** | PBE | 7 (MACE, CHGNet, M3GNet, UMA, M3GNet-MatPES, TensorNet-MatPES, MACE-MatPES) | Main PBE benchmark |
| **MatPES-r2SCAN** | r2SCAN | 3 — **only the r2SCAN-trained FPs** (M3GNet-MatPES-r2SCAN, TensorNet-MatPES-r2SCAN, MACE-MatPES-r2SCAN) | Never mixes in PBE-trained models |
| **OMat24 rattled-1000** | PBE | Same 7 PBE-trained FPs as MatPES-PBE | The PBE-labeled OMat24 rattled-1000 validation dataset — an out-of-distribution robustness check on rattled (randomly displaced) structures, not a separate functional |

Reference-model (`MACE-MatPES`) evaluated-atom counts, current as of the executed analysis
notebooks in this repository:

| Dataset | Evaluated atoms (`\|F_DFT\|>0.01 eV/Å`) | FE fraction (`\|F_DFT\|>1 eV/Å`) |
|---|---|---|
| MatPES-PBE | 3,664,065 | 26.9% |
| MatPES-r2SCAN | 2,870,459 | 33.3% |
| OMat24 rattled-1000 | 1,657,763 | 80.0% |

These are per-FP counts (via `MACE-MatPES`/`MACE-MatPES-r2SCAN` as a representative reference), not
a dataset-wide `reference_population` — the currently-committed `data/*_force_results_standardized.json`
files predate the generators' `reference_population` field (see Section 7 of the generator
notebooks) and don't yet carry one. Structure counts and total-dataset atom counts before any
per-FP filtering are therefore not independently re-verified here; regenerate the standardized
files with the current generator notebooks to populate `reference_population` if you need an
FP-independent dataset-level count.

---

## Benchmark Results

Full per-FP results (MAE/RMSE of `Δ|F|` and `Δθ`, high-accuracy fractions, catastrophic-error
fractions, FE-only statistics) are produced by Section 2 ("Combined Summary Table") of each
analysis notebook and by the interactive figures in [`docs/index.html`](docs/index.html). We
deliberately do **not** publish one overall FP ranking here — the five metrics below are
complementary, not reducible to a single score, and which FP is preferable depends on which
failure mode matters for your use case.

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

`reference_population` is built once from the raw, unfiltered DFT-only data (see the generator
notebooks' Section 7) — it is independent of any one FP's zero-force exclusions, and is what a
dataset-level force distribution or FE fraction should be computed from, not a model-specific
filtered array. Each entry under `models` contains only that FP's own metric-valid atoms, plus
`structure_id`/`atom_index` provenance so they can be traced back to the source dataset.

Raw, per-structure **Cartesian** forces (both DFT and FP) are *not* in the standardized file — they
remain in the generator's per-chunk `results_*.json` checkpoints, together with every atom
(including the ones excluded from the standardized file's metric arrays) and a `null` angle
wherever it's undefined.

---

## Repository Structure

```text
Force_error/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── analysis/
│   ├── force_error_analysis_matpes_pbe.ipynb
│   ├── force_error_analysis_matpes_r2scan.ipynb
│   └── force_error_analysis_omat24_rattled_1000.ipynb
├── generation/
│   ├── matpes_PBE_run_generator.ipynb
│   ├── matpes_r2scan_run_generator.ipynb
│   └── matpes_run_generator_omat24_rattled1000.ipynb
├── scripts/
│   ├── force_results.py          # build_force_results(...) -- the core standardization function
│   ├── force_error_metrics.py    # fraction tables, MAE/RMSE, regime panels, heatmaps
│   ├── fp_cdf_density_plots.py   # CDF computation and 2-D density plotting
│   └── heatmap_table.py          # low-level heatmap drawing primitives
├── examples/
│   └── mace_matpes_cartesian_force_example.json   # small input for the Cartesian-force demo
├── data/
│   └── README.md                 # standardized-file names + how to obtain/regenerate them
└── docs/
    └── index.html                 # results website (open directly or via GitHub Pages)
```

The analysis notebooks were cleared of their heavy plot outputs before publishing — run them
yourself to regenerate all figures (`outputs/` is git-ignored).

---

## Using FPBench with Another Dataset

1. **Analysis only, from paired Cartesian forces.** If you already have DFT and FP Cartesian
   forces (same atom ordering, one `(n_atoms, 3)` array per structure per side), call
   `build_force_results(dft_forces, fp_forces, structure_ids=None)` from `scripts/force_results.py`
   directly — see the worked example at the top of `analysis/force_error_analysis_matpes_pbe.ipynb`.
   It returns the standardized per-atom fields (`dft_force_magnitude`, `fp_force_magnitude`,
   `force_magnitude_error`, `force_angle_error`, `force_vector_error`, `structure_id`, `atom_index`)
   for every model you pass in.
2. **Full dataset generation.** Use one of the three notebooks in `generation/` as a template:
   Section 0 chunks a raw dataset file, Section 2 is a shared `POTENTIAL_REGISTRY` (one entry per
   FP, its own venv/checkpoint), Sections 4–6 generate SLURM scripts and submit them, and Section 7
   performs a strict, cross-model-consistency-checked merge into the standardized schema above.
3. **`structure_id` / `atom_index` are preserved throughout** — from the raw chunk, through the
   per-structure checkpoint, into the final standardized file — so any atom can always be traced
   back to its source structure and position within it.
4. **Configuration.** Every path that is specific to a dataset, cluster, or FP checkpoint is a
   plain variable near the top of the relevant cell (`CHUNK_DATASET_PATH`, `BASE_DATA_DIR`,
   `OUTPUT_DIR`, each registry entry's `venv_activate`/`python`/`model_path`, and the `SLURM`
   dict's `account`/`partition`). Cells raise a clear `ValueError` if you run them before editing
   these placeholders. We recommend a **separate virtual environment per FP family** — their
   PyTorch/ASE/package-version requirements can conflict — and the notebooks never install packages
   automatically.

---

## Python Modules (`scripts/`)

### `force_results.py`
The canonical `build_force_results(dft_forces, fp_forces, structure_ids=None)` — the single
source of truth for how the standardized per-atom fields are computed. The generator notebooks
embed a checksummed, byte-identical reimplementation of its per-structure math so cluster jobs
don't need this module installed.

### `force_error_metrics.py`
Fraction tables (`build_dF_frac_table`, `build_joint_dF_theta_accuracy_table`,
`build_highly_accurate_force_fraction_table`), regime panels (near-eq vs. FE), conditioned
MAE/RMSE (`build_dF_mae_rmse_fdft_subset`, `build_theta_mae_rmse_fdft_subset`), large-error
analysis, and query utilities (`get_bad_atom_indices`).

### `fp_cdf_density_plots.py`
CDF computation (`build_cdf_from_all_results`) and 2-D density panels
(`panel_abs_dF_vs_dtheta_cond_on_Fdft`, `panel_Fdft_vs_abs_dF`, `panel_Fdft_vs_dtheta`) with
PCHIP-smoothed curves and automatically placed inset zooms.

### `heatmap_table.py`
The custom split-triangle heatmap primitives (`draw_triangular_cell`,
`triangular_heatmap_with_fraction_row`, `plot_fraction_panel`) used throughout the notebooks —
each cell can encode two statistics at once (e.g. MAE vs. RMSE, or `Δθ<1°` vs. `Δθ<20°`).

---

## Installation & Quick Start

```bash
git clone https://github.com/kamirian/FPBench.git
cd Force_error
pip install -r requirements.txt

# Place the standardized JSON files in data/ (see data/README.md), then:
jupyter notebook analysis/force_error_analysis_matpes_pbe.ipynb
```

Notebooks use paths relative to their own subdirectory (`../scripts`, `../data`, `../examples`) —
open them from within `analysis/`, or with Jupyter's working directory set there.

---

## Data Access

The standardized JSON files are large (hundreds of MB to a few GB) and are **not** committed to
this repository. See [`data/README.md`](data/README.md) for exact filenames and how to obtain or
regenerate them with the `generation/` notebooks.

---

## Citation

If you use this code or analysis in your work, please cite:

```bibtex
@misc{amirian2025fpbench,
  author = {Kiyan Amirian and Yifei Mo},
  title  = {FPBench: Foundation Potential Force-Error Analysis},
  year   = {2025},
  url    = {https://github.com/kamirian/FPBench}
}
```

---

## License

MIT
