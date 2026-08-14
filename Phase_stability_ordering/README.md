# Phase Stability and Elemental Ordering

The Phase Stability and Elemental Ordering component of [FPBench](../README.md): convex-hull/
relative-phase-stability, elemental-ordering energy ranking, and structural-relaxation (RMSD)
metrics for foundation potentials (FPs).

Leaderboard: **https://kamirian.github.io/FPBench/phase-stability-ordering.html**

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

Minimal public-function example (every name below is verified against
`scripts/convexhull_analysis_utils.py`):

```python
from convexhull_analysis_utils import (
    build_phase_stability_ordering_results,
    validate_phase_stability_ordering_results,
    build_combined_hull_table,
    build_combined_ordering_table,
    build_rmsd_table,
)

benchmark_results = build_phase_stability_ordering_results(
    reference_data=reference_data,   # your own hull/ordering DFT reference
    fp_results=fp_results,           # your own FP relax/static results
)
dft_hull, fp_hull = benchmark_results["dft_hull"], benchmark_results["fp_hull"]
dft_ordering, fp_ordering = benchmark_results["dft_ordering"], benchmark_results["fp_ordering"]

validation_report = validate_phase_stability_ordering_results(
    dft_hull=dft_hull, fp_hull_by_fp=fp_hull,
    dft_ordering=dft_ordering, fp_ordering_by_fp=fp_ordering,
    fps=list(fp_results),
)

hull_table, hull_details = build_combined_hull_table(dft_hull, fp_hull, list(fp_results), model_names)
ordering_table, ordering_summaries = build_combined_ordering_table(dft_ordering, fp_ordering, list(fp_results), model_names)
rmsd_table, rmsd_dfs, rmsd_summaries = build_rmsd_table(dft_hull, fp_hull, list(fp_results), model_names)
```

The small demo (`examples/phase_stability_ordering_demo_data.json`, loaded right after Section 0
of the analysis notebook) runs this exact workflow on one real tie-line system and one real
ordering group, so you can see it work on genuine numbers with no download.

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

| Metric | Direction | Description |
|---|---|---|
| Average energy error (meV/atom) | lower is better | MAE of the FP energy vs. DFT, over all structures in the dataset. |
| Ground-state agreement (%) | higher is better | For each composition, whether the FP-predicted lowest-energy phase agrees with DFT. Pooled across all systems; single-phase compositions (no real choice to get right) are excluded from both numerator and denominator. |
| Within-phase hull-minimum agreement (%) | higher is better | For each phase, whether the FP-predicted minimum-energy composition agrees with DFT (tie-line-normalized energies); single-composition phases excluded. |
| Hull-minimum agreement (%) | higher is better | For each tie-line system, whether the FP-predicted global convex-hull minimum agrees with DFT. |

### Energy rankings of elemental orderings

| Metric | Direction | Description |
|---|---|---|
| Average energy error (meV/atom) | lower is better | MAE of the FP energy vs. DFT, averaged over ordering groups. |
| Top-1 accuracy (%) | higher is better | Fraction of groups where the FP's lowest-energy ordering matches DFT's. |
| Recall@3, Recall@10 (%) | higher is better | Fraction of groups where DFT's ground-truth ordering appears in the FP's top-3 / top-10 by predicted energy. |
| Spearman ρ | higher is better | Rank correlation between FP and DFT energies within each group, averaged. |
| Rate of ranking errors (%) | lower is better | Fraction of pairwise orderings within a group that the FP gets backwards relative to DFT. |
| Mean/max ΔE_DFT of misranked pairs (meV/atom) | lower is better | DFT energy gap for the pairs the FP got backwards -- a large gap misranked is a worse failure than a near-degenerate one. |

### Structural-relaxation effects (RMSD)

Applies to full FP relaxation only (static evaluation does not produce an FP-relaxed structure to
compare). Compares each FP-relaxed structure against its DFT-relaxed counterpart via pymatgen's
`StructureMatcher`.

| Metric | Direction | Description |
|---|---|---|
| Map success (%) | higher is better | Fraction of candidates with an FP result that `StructureMatcher` could successfully map to its DFT counterpart. |
| Mean/max RMSD (Å) | lower is better | RMSD between mapped FP- and DFT-relaxed structures, converted to Å via the geometric mean cell volume. |
| RMSD < 0.05 / 0.10 / 0.20 Å (%) | higher is better | Fraction of mapped candidates below each threshold. |

No overall ranking is published across these three metric groups, or within a group across
unrelated columns; they are complementary, independent measurements. See the paper and Methods
Sections 4.6.1-4.6.3 for complete methodological details.

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
