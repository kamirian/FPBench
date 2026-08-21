# examples/

`ion_migration_neb_demo_data.json` is a small, genuine slice of this
project's own canonical reference and results data, already shaped as one
`{"reference_data": ..., "fp_results": ...}` pair -- the exact input contract
`build_neb_analysis_results(reference_data, fp_results)` accepts (see that
function's docstring in `../scripts/neb_analysis.py`, and Section 0.6 of
`../analysis/neb_analysis.ipynb` for the full schema and required
identifiers). It exists only to demonstrate the public workflow (load ->
validate -> lookup -> barrier/profile analysis -> status handling -> plot)
end to end on real numbers; it is not a manuscript results table and covers
one pathway and one FP, not the full 154-pathway/7-FP benchmark.

## Contents

- `reference_data["pathways"]["29966|8"]`: one full, real DFT-NEB pathway --
  the Oct-Oct Li-migration pathway along the c-axis in Li3YCl6 (ICSD 29966),
  the same pathway shown in the manuscript's representative-path figure.
  All 7 finalized DFT-NEB images, plus the unrelaxed `full_fp_neb_input`
  source endpoint structures.
- `fp_results["models"]["MACE-MP0_medium"]`: MACE's real results for that
  pathway across all three protocols (`full_fp_neb`, `fp_static_on_dft_neb`,
  `dft_static_on_fp_neb`), all `calculation_status: "completed"` and
  `neb_converged: true` in this slice.

In the analysis notebook (Section 0.7) this is loaded and run through
`validate_neb_analysis_inputs(...)` and `build_neb_analysis_results(...)` --
the same functions used for the manuscript-reproduction sections later in
the notebook -- with every result kept in `demo_`-prefixed variables, never
mixed into the manuscript's own analysis variables. A small DFT-vs-FP energy
profile plot for this one pathway is included as the "at least one plot"
demonstration.

## Regenerating

Extracted directly from the private project's approved
`input_data/ion_migration_neb_reference.json` and
`results/ion_migration_neb_results.json` for pathway `29966|8` and FP
`MACE-MP0_medium` only -- every structure, energy, force, identifier, image
ordering, unit, and calculation stage is copied verbatim, never
hand-written or synthesized. No conversion script is checked in for this
one-off extraction; re-extract by loading the two canonical files and
copying `reference_data["pathways"]["29966|8"]` and
`fp_results["models"]["MACE-MP0_medium"]`'s three protocol branches for
that same pathway key.

## The full standardized files (`../data/`)

`ion_migration_neb_reference.json.gz` (154 pathways, shipped in this repo)
and `ion_migration_neb_results_standardized.json` (all seven FPs, not
shipped -- see [`../data/README.md`](../data/README.md) for size, checksum,
and how to obtain it) are the two files the analysis notebook's main
sections actually load for manuscript reproduction. As with the small demo
above, the public analysis contract is always the Python
`reference_data`/`fp_results` structures that `build_neb_analysis_results`
accepts -- `.json`/`.json.gz` are this project's own on-disk convenience
formats, not a requirement placed on external users supplying their own
data.
