# examples/

`phase_stability_ordering_demo_data.json` is a small, genuine slice of this
project's own PCM reference data and MACE results, already shaped as one
`{"reference_data": ..., "fp_results": ...}` pair -- the exact input contract
`build_phase_stability_ordering_results(reference_data, fp_results)` accepts
(see that function's docstring in `convexhull_analysis_utils.py`, Section S,
and the "Using FPBench with another dataset" section of
`convexhull_ordering_analysis_all_models.ipynb` for the full schema and
required identifiers). It exists only to demonstrate the public workflow
(build -> validate -> hull/ordering/RMSD tables) end to end on real numbers;
it is not a manuscript results table and covers a tiny fraction of the full
benchmark -- one tie-line system and one ordering group, not the 22 systems
/ 305 groups used in the manuscript reproduction.

Contents:

- `reference_data["hull"]["Bi2Te3_SiTe2"]`: one full, real tie-line system
  (8 interior candidates + 2 binary endpoints, 10 candidates total) -- the
  smallest of the 22 hull systems used in the manuscript, chosen so the file
  stays small while still being a complete, genuine system (both endmember
  formulas parse with pymatgen, so within-phase/global hull-minimum
  agreement are meaningful for it, not just average error/ground-state
  agreement).
- `reference_data["ordering"]["mp-938_Ge2+1_Bi3+2_Te2-4"]`: one complete
  20-candidate elemental-ordering group.
- `fp_results["mace"]`: MACE's real full-relaxation and static results for
  every candidate/ordering above (all `status: "success"` in this slice).

In the notebook this is loaded and run through
`build_phase_stability_ordering_results(...)`,
`validate_phase_stability_ordering_results(...)`, `build_combined_hull_table(...)`,
`build_combined_ordering_table(...)`, and `build_rmsd_table(...)` -- the same
functions used for the manuscript-reproduction tables in Sections 4-6 -- with
every result kept in its own `demo_`-prefixed variable, never mixed into the
manuscript's `dft_hull`/`fp_hull`/`dft_ordering`/`fp_ordering`.

Regenerate by re-running the extraction against this directory's own loaded
standardized FPBench data structures (`dft_hull["Bi2Te3_SiTe2"]`,
`fp_hull["mace"]`, `dft_ordering["mp-938_Ge2+1_Bi3+2_Te2-4"]`,
`fp_ordering["mace"]`, from Section 1) -- see the "Using FPBench with another
dataset" section of `convexhull_ordering_analysis_all_models.ipynb` for the
exact field mapping.

This demo file is intentionally separate from and unrelated to the two
canonical manuscript-reproduction files under `../data/`
(`phase_stability_ordering_reference.json.gz`,
`phase_stability_ordering_results_standardized.json.gz`) -- those are the
full 22-system / 305-group benchmark for all seven FPs; this is a
hand-picked, illustrative one-system slice for one FP, kept in `examples/`
rather than `data/` for exactly that reason.

## The two canonical standardized files (`../data/`)

`phase_stability_ordering_reference.json.gz` (shared DFT reference: hull +
ordering, endpoints and interior candidates together under each system) and
`phase_stability_ordering_results_standardized.json.gz` (all seven FPs'
results under one `"models"` mapping, keyed by stable model key --
`mace`, `chgnet`, `m3gnet_mp`, `uma`, `m3gnet_matpes_pbe`, `tensornet_pbe`,
`mace_matpes_pbe` -- full relaxation and static evaluation kept separate)
are the two files the notebook's Section 1 actually loads for manuscript
reproduction, via `load_standardized_reference(...)` /
`load_standardized_results(...)` -> `build_phase_stability_ordering_results(...)`.
They are produced once, offline, by
`scripts/convert_legacy_phase_stability_ordering_data.py` from this
project's historical FORMAT A/FORMAT B raw files -- that script and the raw
files it reads are not part of the normal analysis path.

**Adding another FP later**: `merge_phase_stability_ordering_fp_results(...)`
combines one or more standardized single-FP result fragments (the shape
`generation/convexhull_ordering_run_generator.ipynb` produces for one FP) into the merged `"models"`
mapping above, validating each fragment's schema version, dataset name, and
reference checksum against the existing file, and every candidate/ordering
identifier it references against the shared reference -- it never silently
overwrites a model that's already present. See that function's docstring in
`convexhull_analysis_utils.py` (Section T) and the "Using FPBench with
another dataset" -> "Reproducing FPBench from the standardized files" /
"Adding another FP later" subsections of
`convexhull_ordering_analysis_all_models.ipynb` for the exact fragment shape
and a worked call.

As with the small demo above, the public analysis contract is always the
Python `reference_data` / `fp_results` structures -- `.json.gz` is this
project's own on-disk convenience format for its own large dataset, not a
requirement placed on external users.
