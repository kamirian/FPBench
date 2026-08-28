# Phase Stability and Elemental Ordering

The Phase Stability and Elemental Ordering component of [FPBench](../README.md): convex-hull/
relative-phase-stability, elemental-ordering energy ranking, and structural-relaxation (RMSD)
metrics for foundation potentials (FPs).

Leaderboard: **https://mogroupumd.github.io/FPBench/phase-stability-ordering.html**

---

## Using FPBench

There are two ways to use this component:

```text
Provided FPBench reference + new FP calculations
                         or
User DFT reference + user FP results
                          ↓
           standardized reference/results
                          ↓
                validation and analysis
                          ↓
          hull, ordering, and RMSD tables
```

- **Evaluate a new FP on the provided FPBench reference dataset.** Use
  `generation/convexhull_ordering_run_generator.ipynb` to generate per-FP jobs and submission
  scripts against the shipped DFT reference, run them on your cluster, merge the results, then load
  the merged file directly in `analysis/convexhull_ordering_analysis_all_models.ipynb` -- see
  [Generator workflow](#generator-workflow) below and the generator notebook itself for the full
  operational detail.
- **Apply the public analysis functions to another compatible DFT/FP dataset.** Call
  `build_phase_stability_ordering_results(...)` and the table builders in
  `scripts/convexhull_analysis_utils.py` directly with your own data -- see
  [Required inputs and outputs](#required-inputs-and-outputs) and the quick-start example below.

Both routes converge on the same standardized data structures, validator, and table-builder
functions -- nothing about the metrics or analysis code differs between them.

---

## Required inputs and outputs

`build_phase_stability_ordering_results(reference_data, fp_results)` is the entry point for both
routes above:

- **`reference_data`**: your DFT reference, with an optional `"hull"` key (tie-line system →
  candidate → role/phase/composition/energy/structure) and an optional `"ordering"` key
  (ordering group → 20 candidates each with energy/structure).
- **`fp_results`**: your FP's results, keyed by FP name, each with `"hull"`/`"ordering"` →
  `"relax"`/`"static"` → per-candidate `status` (`"success"`/`"missing"`/`"failed"`/
  `"non_converged"`) and `energy_total` (`"relax"` also carries `relaxed_structure`).
- **Returns** `{"dft_hull", "fp_hull", "dft_ordering", "fp_ordering"}`, the exact input shape
  `validate_phase_stability_ordering_results(...)` and the table builders below expect.

See that function's full docstring in `scripts/convexhull_analysis_utils.py` (Section S) for the
complete field-level contract, the worked demonstration in
`analysis/convexhull_ordering_analysis_all_models.ipynb` ("Using FPBench with another dataset"),
[`examples/README.md`](examples/README.md) for a small runnable slice of real data, and
[`data/README.md`](data/README.md) for how this maps onto the on-disk `.json.gz` serialization.

---

## Quick start

```bash
git clone https://github.com/mogroupumd/FPBench.git
cd FPBench/Phase_stability_ordering
pip install -r requirements.txt
pip install jupyterlab
jupyter lab analysis/convexhull_ordering_analysis_all_models.ipynb
```

Minimal public-function example, runnable from the `Phase_stability_ordering/` directory on the
real demo data (every name below is verified against `scripts/convexhull_analysis_utils.py`):

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from convexhull_analysis_utils import (
    build_phase_stability_ordering_results,
    validate_phase_stability_ordering_results,
    build_combined_hull_table,
    build_combined_ordering_table,
    build_rmsd_table,
)

with open("examples/phase_stability_ordering_demo_data.json") as f:
    demo_input = json.load(f)

reference_data = demo_input["reference_data"]
fp_results = demo_input["fp_results"]
fps = list(fp_results)
model_names = {"mace": "MACE"}

benchmark_results = build_phase_stability_ordering_results(
    reference_data=reference_data,
    fp_results=fp_results,
)
dft_hull, fp_hull = benchmark_results["dft_hull"], benchmark_results["fp_hull"]
dft_ordering, fp_ordering = benchmark_results["dft_ordering"], benchmark_results["fp_ordering"]

validation_report = validate_phase_stability_ordering_results(
    dft_hull=dft_hull, fp_hull_by_fp=fp_hull,
    dft_ordering=dft_ordering, fp_ordering_by_fp=fp_ordering,
    fps=fps,
)

hull_table, hull_details = build_combined_hull_table(dft_hull, fp_hull, fps, model_names)
ordering_table, ordering_summaries = build_combined_ordering_table(dft_ordering, fp_ordering, fps, model_names)
rmsd_table, rmsd_dfs, rmsd_summaries = build_rmsd_table(dft_hull, fp_hull, fps, model_names)
```

This is the exact same workflow the analysis notebook runs on this same demo data right after its
Section 0 -- one real tie-line system and one real ordering group, no download required. Replace
`reference_data`/`fp_results`/`fps`/`model_names` with your own to run this against another
dataset.

---

## Files and scripts

| File | What it does |
|---|---|
| `analysis/convexhull_ordering_analysis_all_models.ipynb` | Loads the two standardized files, builds and validates the benchmark data structures, and computes every table on the leaderboard. Also documents the public "Using FPBench with another dataset" workflow. Start here to explore results or reproduce the paper's tables. |
| `generation/convexhull_ordering_run_generator.ipynb` | Generates per-FP job/submission scripts against the standardized DFT reference, and merges completed results into the standardized results file. Use this to evaluate a new FP or extend the benchmark. |
| `scripts/convexhull_analysis_utils.py` | The module both notebooks import: builders, validator, table functions, and metric implementations. See [Public entry points](#public-entry-points-scriptsconvexhull_analysis_utilspy) below. |
| `data/phase_stability_ordering_reference.json.gz` | The standardized DFT reference (hull + ordering). See [Standardized data](#standardized-data). |
| `data/phase_stability_ordering_results_standardized.json.gz` | The standardized, merged results for all seven FPs. See [Standardized data](#standardized-data). |
| `examples/phase_stability_ordering_demo_data.json` | A small, genuine one-system/one-group slice used for the runnable demo in the analysis notebook -- no download needed. See [`examples/README.md`](examples/README.md). |

### Public entry points (`scripts/convexhull_analysis_utils.py`)

`load_standardized_reference`, `load_standardized_results`, `build_phase_stability_ordering_results`,
`validate_phase_stability_ordering_results`, `build_combined_hull_table`,
`build_combined_ordering_table`, `build_rmsd_table`, `merge_phase_stability_ordering_fp_results`,
`PHASE_STABILITY_ORDERING_MODEL_ORDER`, `PHASE_STABILITY_ORDERING_MODEL_NAMES`.

### Generator workflow

`generation/convexhull_ordering_run_generator.ipynb` and the analysis notebook are a
directly-connected pipeline -- the generator's final merged output is exactly what the analysis
notebook loads, with no manual conversion step:

1. The generator loads the standardized DFT reference.
2. You configure one or more FPs and it generates per-FP hull/ordering jobs and submission scripts.
3. You run those jobs on your cluster (the notebook never submits jobs itself).
4. Completed chunks are merged into a per-FP standardized fragment.
5. Fragments are merged into the standardized results file (guarded: disabled by default, so a
   routine "Run All" can never overwrite the canonical results).
6. The analysis notebook loads that merged file directly.

See the generator notebook itself for SLURM configuration, checksum verification, chunk-directory
layout, and failure-handling detail -- these are intentionally not duplicated here.

---

## Standardized data

Two canonical files, both under `data/`:

- **`phase_stability_ordering_reference.json.gz`** -- the shared DFT reference: hull candidates
  (interior and binary-endpoint structures together, per tie-line system) and ordering groups.
- **`phase_stability_ordering_results_standardized.json.gz`** -- all seven FPs' results, keyed by
  stable model key, each with full-relaxation and static-evaluation results kept separate.

The results file records the sha256 of the exact reference file it was merged against; the
analysis notebook verifies this pairing before trusting it. Every candidate is identified by
natural identifiers throughout -- tie-line system name, candidate id, phase id, composition,
ordering-group id, ordering-candidate id, and protocol (`relax`/`static`) -- never an invented or
positional index.

See [`data/README.md`](data/README.md) for the full field-level schema, checksums, and file sizes.

---

## Metrics

### Convex hull and relative phase stability

| Name | Metrics |
|---|---|
| Average energy error | Energy error relative to DFT, reported as the MAE over all structures in the convex-hull benchmark. |
| Ground-state agreement: lowest-energy phase at each composition. | For each composition, the FP-predicted ground-state phase is compared with DFT. The reported metric is the fraction of compositions, across all tie-line systems, for which the two agree. |
| Within-phase hull-minimum agreement: lowest-energy composition within each phase. | For each phase, the FP-predicted minimum-energy composition is compared with DFT. The reported metric is the fraction of phases, across all tie-line systems, for which the two agree. |
| Hull-minimum agreement: global convex-hull minimum across the tie-line. | For each tie-line system, the FP-predicted convex-hull minimum is compared with DFT. The reported metric is the fraction of systems for which the two agree. |

### Energy rankings of elemental orderings

| Name | Metrics |
|---|---|
| Average energy error | Energy error relative to DFT, reported as the MAE over all structures in the ordering benchmark. |
| Top-1 accuracy | Fraction of ordering groups for which the FP identifies the same lowest-energy configuration (LEC) as DFT. |
| Recall@k | Fraction of the k lowest-energy DFT orderings retained among the top k FP-ranked orderings within each ordering group. The reported metric is the mean over all ordering groups. |
| Spearman's ρ | Rank correlation between FP and DFT orderings within each ordering group. The reported metric is the mean over all ordering groups. |
| Rate of ranking errors | Fraction of configuration pairs whose FP and DFT rankings disagree, within each ordering group. The reported metric is the mean over all ordering groups. |
| ΔE<sup>DFT</sup> of misranked pairs | DFT energy difference between configurations that the FP misranks. For each ordering group, the mean and maximum ΔE<sup>DFT</sup> over its misranked pairs are computed and then averaged across all ordering groups. |

### Structural-relaxation effects

Applies to full FP relaxation only (static evaluation does not produce an FP-relaxed structure to
compare).

| Name | Metrics |
|---|---|
| Structure relaxation error | For each structure, the RMSD between the FP-relaxed and DFT-relaxed structures is computed. The reported metric is the mean/maximum over all structures in the dataset. |
| Mean/max RMSD | Geometric deviation after alignment (Å). |
| Map success | Fraction of cases where FP and DFT structures remain sufficiently similar for meaningful RMSD evaluation; failures correspond to large structural deviation. |
| RMSD thresholds | Increasingly strict measures of fidelity. |

No overall ranking is published across these three metric groups, or within a group across
unrelated columns; they are complementary, independent measurements. See the paper and Methods
Sections 4.6.1-4.6.3 for complete methodological details.

---

## Registering a potential

To evaluate an FP that is not already on the leaderboard, add one entry to the
`POTENTIAL_REGISTRY` dictionary in `generation/convexhull_ordering_run_generator.ipynb`. No
metric or analysis code needs changing.

| Field | Purpose |
|---|---|
| dictionary key | Short registry key used for job and script names. Lowercase, filesystem-safe. |
| `mlip_name` | Name written into the standardized results. |
| `site_pkgs` | site-packages directory of the environment this FP runs in. |
| `venv_activate` | Activate script of that environment. |
| `python` | Python executable of that environment. |
| `model_path` | Checkpoint path, or `None` if the package ships its own weights. |
| `calc_setup` | Code that builds the ASE calculator, ending in a variable named `calc`. Refers to the checkpoint as `MODEL_PATH`. |

Copy the shape of an existing entry in the same notebook. Placeholder values beginning
`/path/to/` or `YOUR_` are rejected with a clear error rather than written into a job script.
See [Adding a potential](../ADDING_A_POTENTIAL.md) for the full workflow.

---

## Reproducing the provided benchmark

The two standardized files under `data/` and `analysis/convexhull_ordering_analysis_all_models.ipynb`
together reproduce every result reported in the paper -- running the notebook top to bottom
requires only those files plus `scripts/convexhull_analysis_utils.py`, no cluster or FP packages.
Model versions and official sources for the seven evaluated FPs are documented once on the
[FPBench home page](../README.md#foundation-potentials-evaluated) rather than duplicated here.

---

## Citation and license

The manuscript this work supports is not yet publicly available. Final citation information will
be added here once it is.

MIT license. See the [repository-level LICENSE](../LICENSE).
