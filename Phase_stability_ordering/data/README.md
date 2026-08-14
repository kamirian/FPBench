# Standardized data files

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
[README](../README.md#workflow-generator--submission--merge--analysis) for the full pipeline.
Regenerating is not required just to explore the analysis code or reproduce the published tables:
both files are already here, and the small example in `../examples/` lets you run
`build_phase_stability_ordering_results(...)` and inspect its output immediately, with no
download or cluster access.

## Schema

See the main [README](../README.md#standardized-output-schema) for the full standardized schema
(`schema_version`, `component`, `dataset_name`, `units`, `reference_metadata` /
`generation_metadata`, `reference_data` / `models`).
