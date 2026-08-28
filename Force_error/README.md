# Force Prediction

The Force Prediction component of [FPBench](../README.md): force-magnitude and force-angle error
metrics for foundation potentials (FPs) against DFT reference forces.

Leaderboard: **https://mogroupumd.github.io/FPBench/force-error.html**

---

## Using FPBench

There are two ways to use this component:

```text
Paired Cartesian DFT and FP forces
                  or
Full generator calculations on a dataset
                   ↓
        standardized force_results
                   ↓
          validation and analysis
                   ↓
       FPBench force-error tables
```

- **Build standardized force results directly from paired Cartesian DFT and FP forces.** Call
  `build_force_results(dft_forces, fp_forces, structure_ids=None)` from `scripts/force_results.py`
  -- see [Required inputs and outputs](#required-inputs-and-outputs) and the quick start below.
- **Use a generator notebook as a template for full dataset/cluster generation**, then run the
  matching analysis notebook. Each generator notebook writes SLURM job and submission scripts; it
  does not submit jobs merely by running the generation cells -- submission is always an explicit
  step on your own cluster. See [Files and scripts](#files-and-scripts).

Both routes produce the same standardized `force_results` object that every analysis function
consumes -- nothing about the metrics or analysis code differs between them.

---

## Required inputs and outputs

`build_force_results(dft_forces, fp_forces, structure_ids=None)` is the entry point for the
paired-forces route:

- **`dft_forces`**: one Cartesian DFT force array per structure, shape `(N_i, 3)`, in eV/Å.
- **`fp_forces`**: `dict[str, sequence of array-like]` -- one entry per FP model; each value is a
  per-structure list of Cartesian FP force arrays, matching `dft_forces` in structure count, atom
  count, and atom ordering per structure.
- **`structure_ids`** (optional): one identifier per structure. When given, the returned dicts
  carry `structure_id`/`atom_index` provenance; core analyses do not require it.
- **Returns** `force_results[model]`, each with `dft_force_magnitude`, `fp_force_magnitude`,
  `force_magnitude_error`, `force_angle_error`, `force_vector_error` (and `structure_id`/
  `atom_index` when provided) -- the exact shape the analysis functions in
  `scripts/force_error_metrics.py` and `scripts/fp_cdf_density_plots.py` expect.

See that function's full docstring in `scripts/force_results.py` for the complete field-level
contract, and the worked example in `analysis/force_error_analysis_matpes_pbe.ipynb`.

---

## Quick start

```bash
git clone https://github.com/mogroupumd/FPBench.git
cd FPBench/Force_error
pip install -r requirements.txt
```

Minimal public-function example, runnable from the `Force_error/` directory on the real included
example (every name below is verified against `scripts/force_results.py`):

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from force_results import build_force_results

with open("examples/mace_matpes_cartesian_force_example.json") as f:
    example = json.load(f)

dft_forces = example["dft_forces"]
fp_forces = {"mace": example["fp_forces"]}
structure_ids = example["structure_ids"]

force_results = build_force_results(dft_forces, fp_forces, structure_ids=structure_ids)

print("Models:", list(force_results))
print("Fields:", list(force_results["mace"]))
```

To explore the full leaderboard tables and figures, install `jupyterlab` and open one of the
analysis notebooks:

```bash
pip install jupyterlab
jupyter lab analysis/force_error_analysis_matpes_pbe.ipynb
```

Reproducing the full analysis requires the large standardized JSON files (see
[`data/README.md`](data/README.md)); these are not committed to Git. The small Cartesian-force
example above needs no download.

---

## Files and scripts

| File | What it does |
|---|---|
| `analysis/force_error_analysis_matpes_pbe.ipynb` | Loads the standardized MatPES-PBE results and computes every MatPES-PBE table/figure on the leaderboard. Also has the worked `build_force_results(...)` demonstration. |
| `analysis/force_error_analysis_matpes_r2scan.ipynb` | Same, for the MatPES-r2SCAN dataset (3 r2SCAN-trained FPs). |
| `analysis/force_error_analysis_omat24_rattled_1000.ipynb` | Same, for the OMat24 rattled-1000 validation split. |
| `generation/matpes_PBE_run_generator.ipynb` | Generates per-FP job/submission scripts against the raw MatPES-PBE dataset, and merges completed results into the standardized file. Use as a template to evaluate a new FP on MatPES-PBE. |
| `generation/matpes_r2scan_run_generator.ipynb` | Same, for MatPES-r2SCAN. |
| `generation/matpes_run_generator_omat24_rattled1000.ipynb` | Same, for OMat24 rattled-1000. |
| `scripts/force_results.py` | `build_force_results(...)`, the canonical standardization function -- the module both notebooks and this README's quick start import. |
| `scripts/force_error_metrics.py` | Fraction tables, MAE/RMSE, regime panels, heatmaps, and `get_bad_atom_indices` querying, built from a standardized `force_results` object. |
| `scripts/fp_cdf_density_plots.py` | CDF computation and 2-D density plotting. |
| `scripts/heatmap_table.py` | Low-level heatmap drawing primitives. |
| `examples/mace_matpes_cartesian_force_example.json` | A small, genuine 5-structure slice (real MatPES-PBE DFT forces and real MACE forces) used for the runnable quick start above -- no download needed. |
| `data/*_force_results_standardized.json` | The standardized, merged results for each dataset. Not committed to Git (see [`data/README.md`](data/README.md)). |

The generator notebooks embed a checksummed, byte-identical reimplementation of
`build_force_results(...)`'s per-structure math so cluster jobs do not need `scripts/` installed;
`structure_id`/`atom_index` are preserved throughout, from the raw chunk through the per-structure
checkpoint into the final standardized file, so any atom can be traced back to its source
structure and position within it.

---

## Standardized data

Each dataset has one standardized results file under `data/` (see
[`data/README.md`](data/README.md) for exact filenames, sizes, and how to obtain or regenerate
them). Loading one gives a plain Python dict; analyses consume `data["models"]`, a
`force_results[model]` dict per FP in the schema documented above and in `scripts/force_results.py`.

Raw, per-structure Cartesian forces (both DFT and FP) are not in the standardized file -- they
remain in the generator's per-chunk checkpoints, together with every atom (including ones excluded
from the standardized file's metric arrays).

---

## Metrics

Evaluated for atoms with `|F_DFT| > 0.01 eV/Å` (except the all-atom average-error analysis, which
uses every atom); far-from-equilibrium (FE) atoms are `|F_DFT| > 1 eV/Å`. See the paper for
complete methodological detail.

| Name | Metrics |
|---|---|
| Average error | Force-magnitude error Δ\|F\| and force-angle error Δθ, reported as MAE/RMSE over all atoms or a selected subset. |
| Cumulative distribution functions (CDFs) of force errors | CDFs of \|Δ\|F\|\|, Δθ, and norm of the force-vector error e<sub>vec</sub>, over all atoms or a selected subset. |
| Highly accurate force predictions (small-force-error atoms) | Fraction of atoms with very small values of \|Δ\|F\|\| and Δθ below threshold (e.g. \|Δ\|F\|\| < 0.01 eV/Å). |
| Joint force magnitude-angle accuracy | Fraction of atoms with simultaneously small values of \|Δ\|F\|\| and Δθ (e.g. \|Δ\|F\|\| < 0.01 eV/Å and Δθ < 1° or 20°). |
| Large-force-error atoms | Fraction of atoms with high values of \|Δ\|F\|\| and Δθ (e.g. \|Δ\|F\|\| > 0.5 eV/Å). |
| Force errors on far-from-equilibrium (FE) atoms | MAE/RMSE evaluated for Δ\|F\|, Δθ over the FE atoms selected as \|F<sub>DFT</sub>\| > 1 eV/Å. Fraction of FE atoms with relative force-magnitude error r<sub>F</sub> below increasing thresholds. |

Notation: Δ|F| = |F<sub>FP</sub>| − |F<sub>DFT</sub>| (force-magnitude error); |Δ|F|| for CDFs,
thresholds, and fraction metrics; Δθ, the angle between the FP and DFT force vectors;
e<sub>vec</sub> = ||F<sub>FP</sub> − F<sub>DFT</sub>|| (force-vector error); r<sub>F</sub> =
|Δ|F|| / |F<sub>DFT</sub>| (relative force-magnitude error, FE atoms only).

---

## Registering a potential

To evaluate an FP that is not already on the leaderboard, add one entry to the
`POTENTIAL_REGISTRY` dictionary in `generation/matpes_PBE_run_generator.ipynb` (and the
`matpes_r2scan_` / `matpes_run_generator_omat24_rattled1000` variants for those datasets). No
metric or analysis code needs changing.

| Field | Purpose |
|---|---|
| dictionary key | Short registry key used for job and script names. Lowercase, filesystem-safe. |
| `fp_name` | Name written into the standardized results. |
| `venv_activate` | Activate script of the environment this FP runs in. |
| `python` | Python executable of that environment. |
| `model_path` | Checkpoint path, or `None` if the package ships its own weights. |
| `calc_setup_code` | Code that builds the ASE calculator, ending in a variable named `calc`. Refers to the checkpoint as `MODEL_PATH`. |

Copy the shape of an existing entry in the same notebook. Placeholder values beginning
`/path/to/` or `YOUR_` are rejected with a clear error rather than written into a job script.
See [Adding a potential](../ADDING_A_POTENTIAL.md) for the full workflow.

---

## Reproducing the provided benchmark

The standardized results files under `data/` and the three analysis notebooks together reproduce
every result reported in the paper. See [`data/README.md`](data/README.md) for how to obtain or
regenerate them, and the `generation/` notebooks to evaluate a new FP -- each is a template that
writes SLURM job/submission scripts for your own cluster; running its cells never submits a job.
Once you have results for MatPES-PBE, MatPES-r2SCAN, or OMat24 rattled-1000, open a
[GitHub issue](https://github.com/mogroupumd/FPBench/issues) with the FP's name, architecture,
training data, checkpoint/version, and computed metrics.

Model versions and official sources for the evaluated FPs are documented once on the
[FPBench home page](../README.md#foundation-potentials-evaluated) rather than duplicated here.

---

## Citation and license

The manuscript this work supports is not yet publicly available. Final citation information will
be added here once it is. In the meantime:

```bibtex
@misc{amirian2026fpbench,
  author = {Kiyan Amirian and Ramanuja Srinivasan Saravanan and Felix Adams and Charles E Schwarz and Yifei Mo},
  title  = {FPBench: Foundation Potential Force-Error Analysis},
  year   = {2026},
  url    = {https://github.com/mogroupumd/FPBench}
}
```

MIT license. See the [repository-level LICENSE](../LICENSE).
