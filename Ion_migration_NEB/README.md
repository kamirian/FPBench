# Ion Migration by NEB

The Ion Migration by NEB component of [FPBench](../README.md): migration-barrier and
migration-pathway metrics for foundation potentials (FPs), using the nudged elastic band
(NEB) method.

Leaderboard: **https://mogroupumd.github.io/FPBench/ion-migration-neb.html**

---

## Using FPBench

There are two ways to use this component:

```text
Provided FPBench reference + new FP calculations
                         or
User DFT reference + user FP results
                          |
           standardized reference/results
                          |
                validation and analysis
                          |
     barrier, profile, RMSD, and force-error tables
```

- **Evaluate a new FP on the provided FPBench reference dataset.** Use
  `generation/fp_neb_generation_and_run.ipynb` to generate `full_fp_neb` and
  `fp_static_on_dft_neb` jobs against the shipped DFT reference, run them on your
  cluster, merge the results, then optionally use
  `generation/dft_static_on_fp_neb.ipynb` to generate DFT static diagnostics on your
  FP's own final NEB images. Load the merged file directly in
  `analysis/neb_analysis.ipynb` -- see [Generator workflow](#generator-workflow) below.
- **Apply the public analysis functions to another compatible NEB dataset.** Call
  `build_neb_analysis_results(...)` and the table/metric functions in
  `scripts/neb_analysis.py` directly with your own data -- see
  [Required inputs and outputs](#required-inputs-and-outputs) and the quick-start
  example below.

Both routes converge on the same standardized data structures, validator, and metric
functions -- nothing about the metrics or analysis code differs between them.

---

## What this component evaluates

FPBench evaluates FPs on 154 Li- and Na-ion migration pathways across 109 unique ICSD
structures (a subset of the ion-migration dataset of Saravanan et al., 7 finalized
DFT-NEB images per pathway), using three separate, never-mixed protocols:

| Workflow | Structural input | Results branch |
|---|---|---|
| Full FP-NEB workflow | Unrelaxed source endpoint structures | `full_fp_neb` |
| Static FP evaluations on the DFT-NEB image structures | Finalized DFT-NEB image structures | `fp_static_on_dft_neb` |
| DFT static diagnostics on the final FP-NEB image structures | Final full FP-NEB image structures | `dft_static_on_fp_neb` |

Rather than migration-barrier error alone, the metrics also capture whether FP-NEB
calculations converge and produce a physically meaningful (Normal-Hill) energy profile,
and whether the FP preserves the correct relative endpoint energies. Comparing the full
FP-NEB workflow against static FP evaluations on the DFT-NEB image structures separates
errors observed from static FP evaluations on those images from additional differences
associated with FP endpoint relaxation and FP-NEB pathway optimization.

---

## Required inputs and outputs

`build_neb_analysis_results(reference_data, fp_results, expected_pathways=None, validate=True)`
is the entry point for both routes above:

- **`reference_data`**: your DFT-NEB reference -- `common_pathway_keys` plus, per pathway,
  `identifiers` (`icsd_id`/`source_path_id`), `full_fp_neb_input` (unrelaxed source
  endpoint structures), and `dft_neb_reference.images` (finalized DFT-NEB images: an
  ordered list with `image_index`, `endpoint_role`, `structure`, `energy_total_eV`,
  `forces_eV_per_angstrom`). A pathway is identified throughout by
  `"{icsd_id}|{source_path_id}"`.
- **`fp_results`**: your FP's results, keyed by FP name, each with up to three protocol
  branches (`full_fp_neb`, `fp_static_on_dft_neb`, `dft_static_on_fp_neb`) -> `"pathways"`
  -> per-pathway image records, plus an explicit `neb_status.neb_converged` for every
  `full_fp_neb` pathway (never assumed `true`).
- **Returns** a `NEBAnalysisResults` object exposing every canonical table this project's
  own analysis uses: DFT/FP pathway classification, convergence/status records, endpoint
  RMSD, barrier and energy-profile summaries by protocol, and force-error tables on both
  the FP-NEB and DFT-NEB paths. A protocol with no data for a given FP comes back empty
  or `None`, never fabricated from another protocol.

See that function's full docstring in `scripts/neb_analysis.py`, Section 0.6 of
`analysis/neb_analysis.ipynb` ("Using FPBench data or another NEB dataset") for a worked
example, [`examples/README.md`](examples/README.md) for a small runnable slice of real
data, and [`data/README.md`](data/README.md) for how this maps onto the on-disk JSON
serialization.

---

## Quick start

```bash
git clone https://github.com/mogroupumd/FPBench.git
cd FPBench/Ion_migration_NEB
pip install -r requirements.txt
pip install jupyterlab
jupyter lab analysis/neb_analysis.ipynb
```

Minimal public-function example, runnable from the `Ion_migration_NEB/` directory on the
real demo data (every name below is verified against `scripts/neb_analysis.py`):

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from neb_analysis import (
    validate_neb_analysis_inputs,
    build_neb_analysis_results,
)

with open("examples/ion_migration_neb_demo_data.json") as f:
    demo = json.load(f)

reference_data = demo["reference_data"]
fp_results = demo["fp_results"]
fp_order = list(fp_results["models"])

validation = validate_neb_analysis_inputs(
    reference_data, fp_results, expected_pathways=None, fp_order=fp_order,
)
analysis = build_neb_analysis_results(
    reference_data, fp_results, expected_pathways=None, fp_order=fp_order, validate=True,
)
print(analysis.dft_path_metrics_df)
```

This is the exact same workflow the analysis notebook runs on this same demo data in
Section 0.7 -- one real pathway, one real FP, no download required. Replace
`reference_data`/`fp_results`/`fp_order` with your own to run this against another
dataset.

---

## Files and scripts

| File | What it does |
|---|---|
| `analysis/neb_analysis.ipynb` | Loads the standardized reference/results, validates them, and computes every table and figure on the leaderboard. Also documents the public "Using FPBench data or another NEB dataset" workflow (Section 0.6) and a small runnable example requiring no download (Section 0.7). Start here to explore results or reproduce the paper's tables. |
| `generation/fp_neb_generation_and_run.ipynb` | Generates `full_fp_neb` and `fp_static_on_dft_neb` job/submission scripts against the standardized DFT reference, and merges completed results into a standardized results file. Use this to evaluate a new FP. |
| `generation/dft_static_on_fp_neb.ipynb` | Generates DFT static (VASP, no relaxation) calculations on the final full FP-NEB image structures, and merges parsed results into the `dft_static_on_fp_neb` branch. |
| `scripts/neb_analysis.py` | The module both the analysis notebook and generator notebooks' validation logic build on: loaders, validator, canonical table/metric functions. See [Public entry points](#public-entry-points-scriptsneb_analysispy) below. |
| `scripts/neb_plots.py`, `scripts/heatmap_table.py` | Plotting helpers used by the analysis notebook. |
| `scripts/export_neb_leaderboard.py` | Regenerates `data/ion_migration_neb_leaderboard_summary.json` and its GitHub Pages copy (`../docs/data/...`) from a reference/results file pair, using the same `neb_analysis.py` functions the analysis notebook uses. Run this after regenerating results to keep the leaderboard website in sync. |
| `data/ion_migration_neb_reference.json.gz` | The standardized DFT-NEB reference (154 pathways). See [Standardized data](#standardized-data). |
| `data/ion_migration_neb_leaderboard_summary.json` | The compact per-FP summary the leaderboard website reads directly -- generated deterministically from the same functions as the analysis notebook, never hand-transcribed. |
| `data/ion_migration_neb_results_standardized.json` (not committed) | The standardized, merged results for all seven FPs. See [Standardized data](#standardized-data). |
| `examples/ion_migration_neb_demo_data.json` | A small, genuine one-pathway/one-FP slice used for the runnable demo in the analysis notebook -- no download needed. See [`examples/README.md`](examples/README.md). |

### Public entry points (`scripts/neb_analysis.py`)

`load_neb_datasets`, `validate_neb_analysis_inputs`, `build_neb_analysis_results`,
`build_protocol_coverage_table`, `collect_unsuccessful_records`,
`compute_key_neb_metrics_summary`, `classify_energy_profile`,
`example_canonical_pathway_records`.

### Generator workflow

`generation/fp_neb_generation_and_run.ipynb`, `generation/dft_static_on_fp_neb.ipynb`,
and the analysis notebook are a directly-connected pipeline -- each generator's merged
output is exactly what the analysis notebook loads, with no manual conversion step:

1. `fp_neb_generation_and_run.ipynb` loads the standardized DFT reference.
2. You configure one or more FPs; with `GENERATE_JOBS = True` it generates per-FP
   `full_fp_neb`/`fp_static_on_dft_neb` jobs and, with `WRITE_SUBMISSION_SCRIPTS = True`,
   submission scripts. Both default to `False` so a routine "Run All" never writes
   thousands of files. Before relaxation, each generated `full_fp_neb` job pre-aligns its
   start/end endpoint structures under the minimum-image convention (pymatgen
   `Structure.interpolate(..., pbc=True)`, no site reordering) -- the reference file stores
   each endpoint in its own independent periodic frame, and `matcalc.NEBCalc` interpolates
   with `pbc=False` internally, so without this step a migrating site can get interpolated
   the long way around the cell instead of via its true minimum-image hop. Generated
   `full_fp_neb` jobs use `matcalc.RelaxCalc` for fixed-cell endpoint relaxation and
   `matcalc.NEBCalc` for climbing-image NEB, both driven by an ASE BFGS optimizer that
   MatCalc itself constructs and runs; generated `fp_static_on_dft_neb` jobs are
   calculator-only single-point evaluations, with no relaxation or NEB optimization at all.
3. You run the generated jobs on your cluster (the notebook never submits jobs itself).
4. The notebook's merge step writes one complete candidate results file under its own
   `runs/.../merged/` directory -- never overwriting a canonical results file.
5. Optionally, `dft_static_on_fp_neb.ipynb` reads that (or the canonical) results file,
   generates DFT static calculations on your FP's own final full-NEB images, and merges
   parsed results into a new candidate file's `dft_static_on_fp_neb` branch.
6. Promotion of a validated candidate to the canonical results file is a manual step --
   neither notebook performs it automatically.
7. The analysis notebook loads the (canonical or candidate) results file directly.

See the generator notebooks themselves for SLURM configuration, checksum verification,
chunk-directory layout, and status/failure handling -- these are intentionally not
duplicated here.

---

## Standardized data

- **`data/ion_migration_neb_reference.json.gz`** -- the shared DFT-NEB reference: 154
  active pathways, unrelaxed source endpoint structures (`full_fp_neb_input`) and
  finalized DFT-NEB images (`dft_neb_reference.images`) kept as two separate inputs,
  never substituted for each other.
- **`data/ion_migration_neb_results_standardized.json`** (not committed, ~323 MB -- see
  [`data/README.md`](data/README.md)) -- all seven FPs' results, keyed by stable FP key,
  with `full_fp_neb`, `fp_static_on_dft_neb`, and `dft_static_on_fp_neb` kept separate.

Every pathway/image is identified by natural identifiers throughout -- `icsd_id`,
`source_path_id`, `image_index`, `endpoint_role` -- never an invented or positional
index. See [`data/README.md`](data/README.md) for the full field-level schema,
checksums, and file sizes.

---

## Metrics

Table 8 in the manuscript; implementation details in the Methods.

| Name | Metrics |
|---|---|
| Non-converged paths | Fraction of FP-NEB calculations that do not satisfy the NEB convergence criterion before reaching the maximum number of optimization steps. |
| Barrier error, full FP-NEB | The forward and backward barrier errors are computed relative to DFT for each path. The reported metric is the MAE/RMSE over all converged paths. |
| Barrier error, static FP evaluations on the DFT-NEB images | The same forward and backward barrier errors, computed from static FP evaluations on the DFT-NEB image structures. The reported metric is the MAE/RMSE over all paths. |
| Endpoint energy ranking agreement | Fraction of converged paths for which FP and DFT identify the same lower-energy endpoint between the two endpoints of the migration path, or both classify the two endpoints as equal in energy. |
| Endpoint energy-difference error | For each path, the error in the endpoint energy difference is computed relative to DFT. The reported metric is the MAE/RMSE over all converged paths. |
| Energy-profile shape agreement | Fraction of converged paths for which the FP reproduces the Normal-Hill energy profile, as in DFT-NEBs. |
| Integrated energy-profile difference | For each path, the integrated absolute energy difference between FP and DFT-NEB energy profiles along the normalized reaction coordinate is computed. The reported metric is the MAE/RMSE over all converged paths. |
| Endpoint structure relaxation error | For each endpoint structure, the RMSD between FP-relaxed and DFT-relaxed endpoint structures is computed. The reported metric is the mean/maximum over all endpoint structures of the converged paths. |
| Map success | Fraction of endpoint-structure comparisons, over the converged paths, for which pymatgen's `StructureMatcher` found a valid structural mapping between the FP-relaxed and DFT-relaxed endpoint structure. A failed mapping is excluded from the mean/maximum RMSD, not counted as RMSD = 0. |
| Force errors on FP-NEB path | Mean force-magnitude error \|Δ\|F\|\| and force-angle error Δθ across all atoms for each image structure of the final FP-NEB path. |
| Force errors on DFT-NEB path | Mean force magnitude error \|Δ\|F\|\| and force angle error Δθ across all atoms for each image structure of the final DFT-NEB path. |

The exact denominators (`n_total`/`n_nonconverged`, and which population feeds each MAE/RMSE
pair) are those implemented in `scripts/neb_analysis.py`'s `compute_barrier_error_summaries`
and `compute_profile_summaries` -- not redefined here. No overall ranking is published across
these metrics; they are complementary, independent measurements. See the paper and Methods
Section 4.7 for complete methodological details.

---

## Failure and convergence records

- `calculation_status`: `not_run`, `missing`, `interrupted`, `failed`, `partial`, or
  `completed`. Always separate from scientific convergence -- a completed full FP-NEB
  run can have `neb_status.neb_converged: false`, preserved and reported, never dropped
  or silently treated as failed.
- Failed/missing/interrupted/not-run `full_fp_neb` and `fp_static_on_dft_neb` records
  live in a sibling `unsuccessful_pathways` branch; rejected/failed/missing/not-run
  `dft_static_on_fp_neb` image attempts live in `unsuccessful_image_attempts`. Both are
  read by `neb_analysis.py` for status/coverage reporting only (`build_protocol_coverage_table`,
  `collect_unsuccessful_records`) -- never for any scientific metric, and never changing
  an existing metric's denominator.
- Pathway energy profiles are classified `Normal-Hill`, `Abnormal`, or `Invalid`;
  `Abnormal`/`Invalid` converged paths are preserved and counted (not excluded) for
  energy-profile shape agreement, per the manuscript's definition.

See [`data/README.md`](data/README.md) for the full status-value tables.

---

## Registering a potential

To evaluate an FP that is not already on the leaderboard, add one entry to the
`POTENTIAL_REGISTRY` dictionary in `generation/fp_neb_generation_and_run.ipynb`. No metric or
analysis code needs changing.

| Field | Purpose |
|---|---|
| dictionary key | Short registry key used for job and script names. Lowercase, filesystem-safe. |
| `output_key` | Canonical FP identifier written into the results JSON's `models` dictionary. |
| `display_name` | Name shown in tables and on the leaderboard. |
| `site_pkgs` | site-packages directory of the environment this FP runs in. |
| `venv_activate` | Activate script of that environment. |
| `model_path` | Checkpoint path, or `None` if the package ships its own weights. |
| `import_lines` | Import statements for the calculator. |
| `calc_lines` | Code that builds the ASE calculator, ending in a variable named `calc`. Refers to the checkpoint as `MODEL_PATH`. |

Both `output_key` and the dictionary key are asserted unique, so a collision fails loudly rather
than overwriting another FP's results. Copy the shape of an existing entry in the same notebook.
See [Adding a potential](../ADDING_A_POTENTIAL.md) for the full workflow.

---

## Reproducing the provided benchmark

`data/ion_migration_neb_reference.json.gz` (shipped) plus
`data/ion_migration_neb_results_standardized.json` (see
[`data/README.md`](data/README.md) for how to obtain it) and
`analysis/neb_analysis.ipynb` together reproduce every NEB result reported in the paper.
Model versions and official sources for the seven evaluated FPs are documented once on
the [FPBench home page](../README.md#foundation-potentials-evaluated) rather than
duplicated here.

---

## Citation and license

The manuscript this work supports is not yet publicly available. Final citation
information will be added here once it is.

MIT license. See the [repository-level LICENSE](../LICENSE).
