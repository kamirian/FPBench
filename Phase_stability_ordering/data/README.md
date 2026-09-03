# Standardized data files

> **Archived deposit.** `phase_stability_ordering_reference.json.gz` is also archived under a
> citable DOI at <https://doi.org/10.6084/m9.figshare.33334329>, together with this README. That
> file is the filtered benchmark subset used in this work: the 597-structure convex-hull subset (one lowest-energy
> ordering per phase and composition, across 22 tie-line systems) and the 305 ordering groups of
> 20 candidates each. The underlying DFT-PBE energies for the parent chalcogenide dataset are
> deposited by Adams et al. at <https://doi.org/10.5061/dryad.xd2547dxn> and should be cited as
> the primary source; this record only reorganizes and filters them. The
> deposited copy contains the same reference data as the one in this repository; its metadata block
> differs, because the internal source-file references were removed for the public deposit. Verify
> the repository copy with the SHA-256 below, and the deposited copy with the SHA-256 given in the
> deposit's own README.
> The code that consumes it lives at <https://github.com/mogroupumd/FPBench>. The FP results file
> below is not part of the deposit.

Unlike Force Prediction's multi-gigabyte standardized files, both of this component's files are
small enough to ship directly in the repository -- no external hosting or Git LFS needed.

| Filename | Size | SHA-256 |
|---|---|---|
| `phase_stability_ordering_reference.json.gz` | 7.3 MiB | `1b6addae634f59ec318d31214e17ecd8b8ecdab8a8c0fa9faedf8e7981e4adaa` |
| `phase_stability_ordering_results_standardized.json.gz` | 4.7 MiB | `1b2e74c29d90cf4115bf9426c842a8ac230e811be199548744a94b8ced07f75f` |

`phase_stability_ordering_results_standardized.json.gz` records the sha256 of the exact reference
file it was merged against (`reference.sha256` in the file itself); the analysis notebook
re-verifies this at load time (Section 1), so the pair above is guaranteed self-consistent as
shipped.

## Regenerating

`phase_stability_ordering_results_standardized.json.gz` is produced by
`../generation/convexhull_ordering_run_generator.ipynb`'s Sections 5-9 (generate jobs -> run on
your cluster -> merge chunks into per-FP fragments -> merge fragments into this file), against the
unchanged `phase_stability_ordering_reference.json.gz`. See the main
[README](../README.md#generator-workflow) for the pipeline overview.
Regenerating is not required just to explore the analysis code or reproduce the published tables:
both files are already here, and the small example in `../examples/` lets you run
`build_phase_stability_ordering_results(...)` and inspect its output immediately, with no
download or cluster access.

## Schema

**`phase_stability_ordering_reference.json.gz`** -- the shared DFT reference (hull + ordering,
endpoints and interior candidates together under each system):

```json
{
  "schema_version": "1.0",
  "component": "phase_stability_ordering",
  "dataset_name": "PCM-phase-stability-ordering",
  "units": {"energy": "eV"},
  "reference_metadata": {"reference_population": {"...": "..."}, "...": "..."},
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

**Natural identifiers** (never invented, all present in the files above): tie-line system name,
candidate identifier (unique within its own system for interior candidates, unique among all
endpoints for endpoints), phase identifier, composition identifier, endpoint role and side,
ordering-group identifier, ordering-candidate identifier, and calculation protocol
(`relax` / `static`).

**Status values** (`status` field, both `hull` and `ordering`, both `relax` and `static`):
`"success"` (scored, carries `energy_total` and, for `relax`, `relaxed_structure`), `"missing"`,
`"failed"`, or `"non_converged"` (not scored, but the distinct label is preserved end to end and
reported separately by `validate_phase_stability_ordering_results(...)` rather than being
silently dropped or backfilled from the DFT value). See `build_phase_stability_ordering_results(...)`'s
docstring in `../scripts/convexhull_analysis_utils.py` (Section S) for the complete field-level
contract, including every optional field.
