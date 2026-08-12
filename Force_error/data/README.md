# Standardized data files

The three standardized results files are **not** committed to this repository. They are hundreds
of MB to a few GB each, well beyond what's practical to version-control directly.

| Filename | Approx. size | Contents |
|---|---|---|
| `matpes_pbe_force_results_standardized.json` | ~2.4 GB | 7 FPs evaluated on MatPES-PBE |
| `matpes_r2scan_force_results_standardized.json` | ~835 MB | 3 r2SCAN-trained FPs evaluated on MatPES-r2SCAN |
| `omat24_rattled_1000_force_results_standardized.json` | ~1.1 GB | 7 PBE-trained FPs evaluated on OMat24 rattled-1000 |

## Where to place them

Download (see distribution link, to be added once the files are deposited on Zenodo /
HuggingFace Datasets) or regenerate them, then place them directly in this `data/` directory.
The analysis notebooks load them as `../data/<filename>`.

## Regenerating from scratch

Each file is produced by Section 7 ("Strict All-Model Merge + Standardized Output") of its
corresponding notebook in `../generation/`:

| Standardized file | Generator notebook |
|---|---|
| `matpes_pbe_force_results_standardized.json` | `matpes_PBE_run_generator.ipynb` |
| `matpes_r2scan_force_results_standardized.json` | `matpes_r2scan_run_generator.ipynb` |
| `omat24_rattled_1000_force_results_standardized.json` | `matpes_run_generator_omat24_rattled1000.ipynb` |

This requires the raw dataset, a working environment per FP family, and (for the full datasets)
an HPC cluster. See each generator notebook's Sections 0-3 for configuration. Regenerating is not
required just to explore the analysis code: the small Cartesian-force example in `../examples/`
lets you run `build_force_results(...)` and inspect its output without either the full datasets or
a cluster.

## Schema

See the main [README](../README.md#standardized-output-schema) for the full standardized schema
(`schema_version`, `dataset_name`, `units`, `generation_metadata`, `reference_population`,
`models`).
