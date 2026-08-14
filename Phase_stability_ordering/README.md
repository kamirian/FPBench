# Phase Stability and Elemental Ordering

The Phase Stability and Elemental Ordering component of [FPBench](../README.md). Evaluates
foundation potentials (FPs) on convex-hull/relative-phase-stability prediction, elemental-ordering
energy ranking, and structural-relaxation accuracy, on a phase-change-material (PCM) tie-line
dataset.

Leaderboard: **https://kamirian.github.io/FPBench/phase-stability-ordering.html**

---

## Benchmark and dataset

The dataset covers 22 ternary tie-line systems built from binary PCM endmembers (e.g.
`GeTe_Sb2Te3`), each with:

- **Interior candidates**: DFT-PBE-relaxed structures along the tie-line between the two binary
  endpoints, 561 total.
- **Binary endpoints**: 36 unique endpoint structures. An endpoint bordering more than one system
  is the same physical structure and is computed once per FP, then shared by every system it
  borders -- 679 total per-system hull records (561 interior + 118 endpoint appearances) once
  endpoints are distributed to their systems, but only 597 unique candidates (561 + 36) are ever
  actually computed.
- **Elemental-ordering groups**: 305 groups (one per tie-line composition selected for ordering
  analysis), each with exactly 20 distinct atomic-ordering candidates at that composition --
  6,100 ordering records total.

All DFT energies are raw, uncorrected VASP-PBE energies -- no Materials Project MP2020
compatibility correction is applied anywhere, for either the binary endpoints or the interior
tie-line structures.

## Two protocols, evaluated separately

- **Full FP relaxation**: the FP relaxes each structure itself, starting from the DFT
  `initial_structure`. Atomic positions are relaxed with ASE `FIRE`
  (`fmax = 0.01 eV/Å`, up to 10,000 steps); cell shape and volume are held fixed throughout (a
  fixed-cell relaxation, not a variable-cell one). Requires the FP's own energy *and* its own
  relaxed structure for every candidate -- used for the RMSD comparison and for the within-phase /
  global hull-minimum metrics, which recompute each candidate's tie-line fraction from that
  structure.
- **Static FP evaluation on DFT-relaxed structures**: the FP evaluates the energy of the DFT
  `relaxed_structure` without relaxing it. Requires only the FP's energy; no separate FP structure
  is needed or used, and RMSD does not apply to this protocol.

A candidate a FP has a result for is recorded with `status: "success"` and its `energy_total`; one
it does not is `status: "missing"`, `"failed"`, or `"non_converged"` (or simply omitted, treated
the same as `"missing"`). Only `status: "success"` records are ever scored, but the distinct label
is preserved and reported separately by the validator -- a missing/failed/non-converged result is
never silently dropped or backfilled from the DFT value.

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

Applies to **full FP relaxation only** (static evaluation does not produce an FP-relaxed
structure to compare). Compares each FP-relaxed structure against its DFT-relaxed counterpart via
pymatgen's `StructureMatcher` (`ltol=0.5`, `stol=0.5`, `angle_tol=10°`), for every unique intended
candidate (endpoints deduplicated by stable identity, not counted once per system they border).

| Metric | Direction | Description |
|---|---|---|
| Map success (%) | higher is better | Fraction of candidates with an FP result that `StructureMatcher` could successfully map to its DFT counterpart. |
| Mean/max RMSD (Å) | lower is better | RMSD between mapped FP- and DFT-relaxed structures, converted to Å via the geometric mean cell volume. |
| RMSD < 0.05 / 0.10 / 0.20 Å (%) | higher is better | Fraction of mapped candidates below each threshold. |

The map-success denominator includes every intended candidate, distinguishing **calculation
failure** (the FP result itself is not `status: "success"` -- no structure exists to compare) from
**mapping failure** (an FP structure exists but `StructureMatcher` could not map it).

No overall ranking is published across these three metric groups, or within a group across
unrelated columns; they are complementary, independent measurements.

---

## Standardized Output Schema

Two files, both under `data/`:

**`phase_stability_ordering_reference.json.gz`** -- the shared DFT reference (hull + ordering,
endpoints and interior candidates together under each system):

```json
{
  "schema_version": "1.0",
  "component": "phase_stability_ordering",
  "dataset_name": "PCM-phase-stability-ordering",
  "units": {"energy": "eV"},
  "reference_metadata": {"reference_population": {"n_systems": 22, "n_interior_candidates": 561, "n_unique_endpoints": 36, "n_total_hull_candidates": 597, "n_ordering_groups": 305, "n_ordering_records": 6100}, "...": "..."},
  "reference_data": {
    "hull": {
      "<tie_line_system>": {
        "<candidate_id>": {
          "role": "interior | endpoint", "phase_id": "...", "composition": "...",
          "energy_total": 0.0, "relaxed_structure": {"...": "pymatgen Structure.as_dict()"},
          "initial_structure": {"...": "..."}, "endpoint_side": "left | right"
        }
      }
    },
    "ordering": {
      "<ordering_group_id>": {
        "system": "...", "phase_id": "...", "composition": "...",
        "orderings": {
          "<ordering_candidate_id>": {
            "energy_total": 0.0, "relaxed_structure": {"...": "..."}, "initial_structure": {"...": "..."}
          }
        }
      }
    }
  }
}
```

**`phase_stability_ordering_results_standardized.json.gz`** -- all seven FPs' results, keyed by
stable model key:

```json
{
  "schema_version": "1.0",
  "component": "phase_stability_ordering",
  "dataset_name": "PCM-phase-stability-ordering",
  "units": {"energy": "eV"},
  "reference": {"sha256": "..."},
  "generation_metadata": {"...": "..."},
  "models": {
    "<model_key>": {
      "metadata": {"registry_key": "...", "mlip_name": "...", "model_path": "...", "protocol": {"...": "..."}, "counts": {"...": "..."}},
      "hull": {"relax": {"<system>": {"<candidate_id>": {"status": "success|missing|failed|non_converged", "energy_total": 0.0, "relaxed_structure": {"...": "..."}}}}, "static": {"...": "same shape, no relaxed_structure"}},
      "ordering": {"relax": {"<ordering_group_id>": {"<ordering_candidate_id>": {"status": "...", "energy_total": 0.0, "relaxed_structure": {"...": "..."}}}}, "static": {"...": "same shape, no relaxed_structure"}}
    }
  }
}
```

`reference.sha256` records the sha256 of the exact `phase_stability_ordering_reference.json.gz`
this results file was merged against; the analysis notebook verifies this before trusting the pair
(Section 1). Model keys, in canonical order: `mace`, `chgnet`, `m3gnet_mp`, `uma`,
`m3gnet_matpes_pbe`, `tensornet_pbe`, `mace_matpes_pbe`.

**Natural identifiers** (never invented, all present in the reference/results files above):
tie-line system name, candidate identifier (unique within its own system for interior candidates,
unique among all endpoints for endpoints), phase identifier, composition identifier, endpoint role
and side, ordering-group identifier, ordering-candidate identifier, and calculation protocol
(`relax` / `static`).

---

## Repository Structure

```text
Phase_stability_ordering/
├── README.md
├── analysis/
│   └── convexhull_ordering_analysis_all_models.ipynb
├── generation/
│   └── convexhull_ordering_run_generator.ipynb
├── scripts/
│   └── convexhull_analysis_utils.py
├── examples/
│   ├── README.md
│   └── phase_stability_ordering_demo_data.json
├── data/
│   ├── README.md
│   ├── phase_stability_ordering_reference.json.gz
│   └── phase_stability_ordering_results_standardized.json.gz
└── requirements.txt
```

---

## Workflow: generator → submission → merge → analysis

`generation/convexhull_ordering_run_generator.ipynb` and `analysis/convexhull_ordering_analysis_all_models.ipynb`
are a directly-connected pipeline: the generator's final merged output is exactly what the
analysis notebook loads, with **no manual conversion step** in between.

1. **Load the standardized DFT reference** (generator Section 1) -- validates every required
   identifier/energy/structure is present before generating anything; stops rather than inventing
   a substitute if something is missing.
2. **Configure one or more FPs** (generator Sections 2-3) -- edit `POTENTIAL_REGISTRY` for the
   FP(s) you intend to run (see "Configuring your own FP" below) and set `SELECTED_POTENTIALS`.
3. **Generate hull and ordering jobs** (Sections 5-6) -- one chunked Python job script + SLURM
   script per chunk, under `OUTPUT_ROOT/{fp}/{hull,ordering}/chunk_XXXX/`. Each chunk carries its
   own `chunk_meta.json` (reference checksum, model key, task, chunk id, calculator-setup
   checksum) that the merge step later verifies before trusting that chunk's results.
4. **Generate per-model and all-model submission scripts** (Section 7) -- one `submit_model.sh`
   per FP (submits every chunk for that FP with `sbatch --parsable`, run from inside each chunk's
   own directory) and one top-level `submit_all_models.sh`. **The notebook never calls `sbatch`
   itself** -- these are shell scripts written to disk for you to run manually on your cluster.
5. **Run jobs externally**, on your cluster.
6. **Merge completed chunks into per-FP fragments** (Section 8) -- `write_fp_fragment(fp_key)`
   reads every chunk's `chunk_results.json`, verifies its embedded `chunk_metadata` against the
   currently-loaded reference and the requested FP (rejecting a stale or mismatched chunk outright
   rather than trusting it under the current checksum), fans each shared endpoint result out to
   every system it borders, and writes `OUTPUT_ROOT/{fp}/standardized_model_fragment.json.gz`.
   Raises if any candidate's chunk output is still missing -- it will not write a fragment from a
   partial run.
7. **Merge fragments into the final standardized results file** (Section 9, guarded) --
   `merge_all_fp_fragments(...)` combines the selected fragments via
   `merge_phase_stability_ordering_fp_results(...)` into
   `data/phase_stability_ordering_results_standardized.json.gz`. This section is disabled by
   default (`RUN_REAL_MERGE = False`) so a routine "Run All" can never overwrite the canonical
   results; enabling it requires explicitly setting `RUN_REAL_MERGE = True` and choosing
   `MERGE_MODE = "fresh_build"` (refuses if the merged file already exists) or `"add_new_fp"`
   (refuses if it does not exist yet) -- `merge_phase_stability_ordering_fp_results` itself still
   rejects any duplicate model key on top of that.
8. **Load the result directly in the analysis notebook** (analysis Section 1) --
   `load_standardized_reference(...)` / `load_standardized_results(...)` ->
   `build_phase_stability_ordering_results(...)`, exactly as shipped.

---

## Reproducing the provided benchmark

```bash
git clone https://github.com/mogroupumd/FPBench.git
cd FPBench/Phase_stability_ordering
pip install -r requirements.txt
pip install jupyterlab
jupyter lab analysis/convexhull_ordering_analysis_all_models.ipynb
```

The two standardized files under `data/` are the exact ones the manuscript results were computed
from -- running the analysis notebook top to bottom reproduces every table on the leaderboard,
using only these two files plus `scripts/convexhull_analysis_utils.py`, no cluster or FP packages
required. See [`data/README.md`](data/README.md) for their checksums and schema pointer.

The small demo (`examples/phase_stability_ordering_demo_data.json`, loaded in the analysis
notebook right after Section 0) runs the same `build_phase_stability_ordering_results(...)` ->
`validate_phase_stability_ordering_results(...)` -> table-builder workflow on one real tie-line
system and one real ordering group, so you can see the whole pipeline work on genuine numbers
without any download. See [`examples/README.md`](examples/README.md).

---

## Configuring and evaluating another FP

1. **Set up your FP's environment.** Clone the repository and install your FP's own package in a
   separate environment, alongside `requirements.txt`. The generator notebook never installs
   packages automatically.
2. **Edit `POTENTIAL_REGISTRY`** in `generation/convexhull_ordering_run_generator.ipynb` (Section
   2) for your FP: `site_pkgs`, `venv_activate`, `python`, `model_path` (or `None` if your
   calculator loads its default checkpoint from the package itself, as CHGNet's does), and
   `calc_setup` (the calculator-construction code, embedded verbatim into every generated job
   script). Every path in the shipped registry is a `/path/to/...` placeholder --
   `_require_configured(...)` raises a clear error if you try to generate jobs for an FP whose
   registry entry, or the SLURM `account`/`partition` in Section 3, is still unedited.
3. **Add your FP's key** to `SELECTED_POTENTIALS` (Section 3), then run Sections 5-7 to generate
   its jobs and submission scripts.
4. **Run the generated jobs** on your cluster (Section 7's scripts, run manually -- the notebook
   never submits jobs itself).
5. **Merge and finalize** with Sections 8-9, following the guarded `RUN_REAL_MERGE` workflow above.
6. **Recompute the metrics** by re-running `analysis/convexhull_ordering_analysis_all_models.ipynb`
   against the updated `data/phase_stability_ordering_results_standardized.json.gz` -- your FP now
   appears in every table exactly like the seven already there.
7. **Open a GitHub issue** with your FP's name, architecture, training data, checkpoint/version,
   and computed metrics: https://github.com/mogroupumd/FPBench/issues

---

## Using FPBench with your own data (not this project's tie-line dataset)

You are not required to use this project's file layout or the two standardized `.json.gz` files at
all. The public analysis contract is the plain Python `reference_data` / `fp_results` structures
`build_phase_stability_ordering_results(reference_data, fp_results)` accepts -- see its full
docstring in `scripts/convexhull_analysis_utils.py` (Section S) and the "Using FPBench with
another dataset" section of `analysis/convexhull_ordering_analysis_all_models.ipynb` for the
complete schema, worked calls, and the exact required/optional fields.

```python
from convexhull_analysis_utils import (
    build_phase_stability_ordering_results,
    validate_phase_stability_ordering_results,
    build_combined_hull_table,
    build_combined_ordering_table,
    build_rmsd_table,
    merge_phase_stability_ordering_fp_results,
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

`merge_phase_stability_ordering_fp_results(...)` combines one or more standardized single-FP
result fragments into the merged `{"models": {...}}` structure `build_phase_stability_ordering_results`
accepts as `fp_results` -- the same function the generator's Section 9 uses, available for anyone
building their own merge workflow around their own data.

These functions never validate against, or fall back to, this project's own historical raw file
layout; `reference_data`/`fp_results` (or their `.json.gz`-serialized form, via
`load_standardized_reference`/`load_standardized_results`) are the entire public interface.

---

## Foundation Potentials Evaluated

Model versions and official sources -- see the [FPBench home page](../README.md#foundation-potentials-evaluated)
for full checkpoint/training-dataset details (shared across all FPBench components).

| FP | Model & version | Official source |
|---|---|---|
| MACE | ≥v0.3.10 (MACE-MPA-0, medium) | [MACE foundation models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) |
| CHGNet | v0.3.0 | [CHGNet (GitHub)](https://github.com/CederGroupHub/chgnet) |
| M3GNet | MP-2021.2.8-PES | [MatGL](https://matgl.ai/) |
| UMA | s-1p1 | [FAIR Chemistry](https://fair-chem.github.io/) |
| M3GNet-MatPES | v2025.1 | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| TensorNet-MatPES | v2025.1 | [MatGL](https://matgl.ai/) &middot; [MatPES](https://matpes.ai/) |
| MACE-MatPES | ≥v0.3.10 (fine-tuned on MatPES-PBE) | [MACE (GitHub)](https://github.com/acesuit/mace) |

---

## Python Module (`scripts/`)

**`convexhull_analysis_utils.py`** -- everything above is implemented here. Key public entry
points: `load_standardized_reference`, `load_standardized_results`,
`build_phase_stability_ordering_results`, `validate_phase_stability_ordering_results`,
`build_combined_hull_table`, `build_combined_ordering_table`, `build_rmsd_table`,
`merge_phase_stability_ordering_fp_results`, `PHASE_STABILITY_ORDERING_MODEL_ORDER`,
`PHASE_STABILITY_ORDERING_MODEL_NAMES`.

---

## Citation

The manuscript this work supports is not yet publicly available. Final citation information will
be added here once it is.

---

## License

MIT. See the [repository-level LICENSE](../LICENSE).
