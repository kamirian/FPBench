"""Synthetic tests for neb_analysis.py's reusable public API
(build_neb_analysis_results, validate_neb_analysis_inputs) plus the
example-record template. Runs against small, in-memory synthetic data --
requires no personal paths and no downloaded FPBench data files.

Run with: python tests/test_neb_analysis.py (from the Ion_migration_NEB/ directory)
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import neb_analysis as na


def _base_dataset():
    """One valid complete pathway (protocol full_fp_neb + fp_static_on_dft_neb
    + dft_static_on_fp_neb) plus one failed/non-converged full_fp_neb
    pathway, built from example_canonical_pathway_records()."""
    ex = na.example_canonical_pathway_records()
    reference_data = {
        "schema_version": "1.0.0",
        "component": "ion_migration_neb",
        "dataset_name": "synthetic test reference",
        "units": {"energy": "eV", "forces": "eV/angstrom"},
        "common_pathway_keys": ["999001|1"],
        "pathways": {"999001|1": ex["dft_reference_pathway"]},
    }
    fp_results = {
        "schema_version": "1.0.0",
        "component": "ion_migration_neb",
        "dataset_name": "synthetic test results",
        "units": {"energy": "eV", "forces": "eV/angstrom"},
        "models": {
            "MyFP": {
                "metadata": {"fp_key": "MyFP", "display_name": "MyFP"},
                "full_fp_neb": {"pathways": {"999001|1": ex["full_fp_neb_pathway"]}},
                "fp_static_on_dft_neb": {"pathways": {"999001|1": ex["fp_static_on_dft_neb_pathway"]}},
                "dft_static_on_fp_neb": {"pathways": {"999001|1": ex["dft_static_on_fp_neb_pathway"]}},
            }
        },
    }
    return reference_data, fp_results, ex


results = []


def check(name, condition):
    results.append((name, bool(condition)))
    print(("PASS" if condition else "FAIL") + f"  {name}")


# ── 1. one valid complete pathway ───────────────────────────────────────────
ref, fp, ex = _base_dataset()
report = na.validate_neb_analysis_inputs(ref, fp, expected_pathways=None, fp_order=["MyFP"])
check("1. valid complete pathway: validation report is ok", report.ok)

analysis = na.build_neb_analysis_results(ref, fp, expected_pathways=None, validate=True)
check("1. build_neb_analysis_results returns dft_path_metrics_df with 1 row",
      len(analysis.dft_path_metrics_df) == 1)
check("1. fp_path_metrics_by_protocol has both protocols for MyFP",
      set(analysis.fp_path_metrics_by_protocol["MyFP"].keys()) == {"full_fp_neb", "fp_static_on_dft_neb"})
check("1. full_fp_neb_path_force_errors_df is present (dft_static_on_fp_neb data given)",
      analysis.full_fp_neb_path_force_errors_df is not None and len(analysis.full_fp_neb_path_force_errors_df) == 3)
check("1. dft_neb_path_force_errors_df is present (fp_static_on_dft_neb data given)",
      analysis.dft_neb_path_force_errors_df is not None and len(analysis.dft_neb_path_force_errors_df) == 3)
check("1. coverage_mismatch_df is empty (image counts match)",
      len(analysis.coverage_mismatch_df) == 0)

# ── 2. reordered images ─────────────────────────────────────────────────────
ref2, fp2, _ = _base_dataset()
imgs = ref2["pathways"]["999001|1"]["dft_neb_images"]
imgs[0], imgs[1] = imgs[1], imgs[0]  # swap so image_index sequence is no longer 0..N-1 in order
imgs[0]["image_index"], imgs[1]["image_index"] = imgs[1]["image_index"], imgs[0]["image_index"]
report2 = na.validate_neb_analysis_inputs(ref2, fp2, fp_order=["MyFP"])
check("2. reordered images: validation catches bad ordering", not report2.ok)

# ── 2b. duplicate images ────────────────────────────────────────────────────
ref2b, fp2b, _ = _base_dataset()
ref2b["pathways"]["999001|1"]["dft_neb_images"].append(
    copy.deepcopy(ref2b["pathways"]["999001|1"]["dft_neb_images"][1]))
report2b = na.validate_neb_analysis_inputs(ref2b, fp2b, fp_order=["MyFP"])
check("2b. duplicate image (index repeated): validation catches bad ordering", not report2b.ok)

# ── 3. mismatched endpoints (last image not tagged 'final') ────────────────
ref3, fp3, _ = _base_dataset()
ref3["pathways"]["999001|1"]["dft_neb_images"][-1]["endpoint_role"] = "intermediate"
report3 = na.validate_neb_analysis_inputs(ref3, fp3, fp_order=["MyFP"])
check("3. mismatched endpoint tag: validation catches it", not report3.ok)

# ── 4. wrong energy units (declared units.energy != 'eV') ──────────────────
ref4, fp4, _ = _base_dataset()
ref4["units"]["energy"] = "eV/atom"
report4 = na.validate_neb_analysis_inputs(ref4, fp4, fp_order=["MyFP"])
check("4. wrong declared energy units: validation catches it", not report4.ok)

# ── 5. wrong force-array shape (forces length != atom count) ───────────────
ref5, fp5, _ = _base_dataset()
ref5["pathways"]["999001|1"]["dft_neb_images"][0]["forces_eV_per_angstrom"] = [[0.0, 0.0, 0.0]]  # 1 atom, not 2
report5 = na.validate_neb_analysis_inputs(ref5, fp5, fp_order=["MyFP"])
check("5. wrong force-array shape: validation catches it", not report5.ok)

# ── 6. missing structures/forces ────────────────────────────────────────────
ref6, fp6, _ = _base_dataset()
del ref6["pathways"]["999001|1"]["dft_neb_images"][1]["forces_eV_per_angstrom"]
try:
    report6 = na.validate_neb_analysis_inputs(ref6, fp6, fp_order=["MyFP"])
    check("6. missing forces field: validation catches it or raises cleanly", not report6.ok)
except Exception as e:
    check(f"6. missing forces field: raised {type(e).__name__} instead of silently proceeding", True)

# ── 7. failed / non-converged full FP-NEB pathway ───────────────────────────
ref7, fp7, ex7 = _base_dataset()
ref7["pathways"]["999002|1"] = copy.deepcopy(ex7["dft_reference_pathway"])
ref7["pathways"]["999002|1"]["identifiers"]["icsd_id"] = "999002"
ref7["common_pathway_keys"].append("999002|1")
fp7["models"]["MyFP"]["full_fp_neb"]["pathways"]["999002|1"] = ex7["failed_full_fp_neb_pathway"]
analysis7 = na.build_neb_analysis_results(ref7, fp7, validate=True)
non_conv_status = analysis7.full_fp_neb_status_by_fp_path["MyFP"].get(("999002", "1"), {})
check("7. non-converged pathway preserved with neb_converged=False (not dropped, not defaulted True)",
      non_conv_status.get("neb_converged") is False)
conv_summary = analysis7.full_fp_neb_convergence_summary["MyFP"]["full_fp_neb"]
check("7. non-converged pathway counted in full_fp_neb_convergence_summary n_neb_not_conv",
      conv_summary["n_neb_not_conv"] >= 1)

# ── 8. protocol-tag mismatch: fp_static_on_dft_neb image carries its own 'structure' ──
ref8, fp8, _ = _base_dataset()
fp8["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]["0"]["structure"] = \
    ref8["pathways"]["999001|1"]["dft_neb_images"][0]["structure"]
report8 = na.validate_neb_analysis_inputs(ref8, fp8, fp_order=["MyFP"])
check("8. protocol-tag mismatch (fp_static_on_dft_neb carrying its own structure): validation catches it",
      not report8.ok)

# ── 8b. full_fp_neb missing its own structure (protocol mixing the other direction) ──
ref8b, fp8b, _ = _base_dataset()
del fp8b["models"]["MyFP"]["full_fp_neb"]["pathways"]["999001|1"]["final_fp_neb_images"]["0"]["structure"]
report8b = na.validate_neb_analysis_inputs(ref8b, fp8b, fp_order=["MyFP"])
check("8b. full_fp_neb image missing its own structure: validation catches it", not report8b.ok)

# ── 9. dataset with fewer/more pathways than expected ──────────────────────
ref9, fp9, _ = _base_dataset()
report9 = na.validate_neb_analysis_inputs(ref9, fp9, expected_pathways=155, fp_order=["MyFP"])
check("9. expected_pathways=155 against a 1-pathway dataset: validation flags the count mismatch",
      not report9.ok)
report9b = na.validate_neb_analysis_inputs(ref9, fp9, expected_pathways=None, fp_order=["MyFP"])
check("9b. expected_pathways=None: no pathway-count assumption forced (does not itself fail this check)",
      not any(c["check"].startswith("DFT reference pathway count ==") for c in report9b.checks))

# ── protocols-available example: only fp_static_on_dft_neb supplied ────────
ref10, fp10, ex10 = _base_dataset()
fp10_partial = {
    "models": {
        "MyFP": {
            "metadata": {"fp_key": "MyFP"},
            "full_fp_neb": {"pathways": {}},
            "fp_static_on_dft_neb": {"pathways": {"999001|1": ex10["fp_static_on_dft_neb_pathway"]}},
            "dft_static_on_fp_neb": {"pathways": {}},
        }
    }
}
analysis10 = na.build_neb_analysis_results(ref10, fp10_partial, validate=True)
check("10. only fp_static_on_dft_neb supplied: dft_neb_path_force_errors_df present",
      analysis10.dft_neb_path_force_errors_df is not None and len(analysis10.dft_neb_path_force_errors_df) == 3)
check("10. only fp_static_on_dft_neb supplied: full_fp_neb_path_force_errors_df is None (not fabricated)",
      analysis10.full_fp_neb_path_force_errors_df is None)

# ── 11. missing schema_version / component / dataset_name (Section F) ──────
for field in ("schema_version", "component", "dataset_name"):
    ref11, fp11, _ = _base_dataset()
    del ref11[field]
    report11 = na.validate_neb_analysis_inputs(ref11, fp11, fp_order=["MyFP"])
    check(f"11. DFT reference missing '{field}': validation catches it", not report11.ok)
    ref11b, fp11b, _ = _base_dataset()
    del fp11b[field]
    report11b = na.validate_neb_analysis_inputs(ref11b, fp11b, fp_order=["MyFP"])
    check(f"11. FP results missing '{field}': validation catches it", not report11b.ok)

# ── 12. missing reference units / missing FP-result units (Section H) ──────
ref12, fp12, _ = _base_dataset()
del ref12["units"]
report12 = na.validate_neb_analysis_inputs(ref12, fp12, fp_order=["MyFP"])
check("12. DFT reference missing units entirely: validation catches it", not report12.ok)

ref12b, fp12b, _ = _base_dataset()
del fp12b["units"]
report12b = na.validate_neb_analysis_inputs(ref12b, fp12b, fp_order=["MyFP"])
check("12. FP results missing units entirely: validation catches it", not report12b.ok)

# ── 13. mismatched DFT vs FP units (Section H) ──────────────────────────────
ref13, fp13, _ = _base_dataset()
fp13["units"]["energy"] = "eV/atom"
report13 = na.validate_neb_analysis_inputs(ref13, fp13, fp_order=["MyFP"])
check("13. mismatched DFT/FP units.energy: validation catches the cross-file disagreement", not report13.ok)
check("13. per-file unit checks (DFT side) still individually pass when only FP's is wrong",
      any(c["check"] == "DFT reference declares units.energy == 'eV'" and c["ok"] for c in report13.checks))

# ── 14. same image count but different image-index sets (Section I) ────────
ref14, fp14, _ = _base_dataset()
static_imgs = fp14["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]
static_imgs["5"] = static_imgs.pop("2")  # same count (3), but indices are now {0,1,5} not {0,1,2}
report14 = na.validate_neb_analysis_inputs(ref14, fp14, fp_order=["MyFP"])
check("14. fp_static_on_dft_neb same count, different index set: validation catches it (not just counting)",
      not report14.ok)

# ── 15. missing static image / 16. unexpected static image (Section I) ─────
ref15, fp15, _ = _base_dataset()
del fp15["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]["2"]
report15 = na.validate_neb_analysis_inputs(ref15, fp15, fp_order=["MyFP"])
check("15. fp_static_on_dft_neb missing image index 2: validation catches it", not report15.ok)

ref16, fp16, _ = _base_dataset()
fp16["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]["9"] = \
    copy.deepcopy(fp16["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]["0"])
report16 = na.validate_neb_analysis_inputs(ref16, fp16, fp_order=["MyFP"])
check("16. fp_static_on_dft_neb unexpected extra image index 9: validation catches it", not report16.ok)

# ── 17. dft_static_on_fp_neb structure from the wrong pathway (Section I) ──
ref17, fp17, ex17 = _base_dataset()
ref17["pathways"]["999002|1"] = copy.deepcopy(ex17["dft_reference_pathway"])
ref17["pathways"]["999002|1"]["identifiers"]["icsd_id"] = "999002"
fp17["models"]["MyFP"]["full_fp_neb"]["pathways"]["999002|1"] = copy.deepcopy(ex17["full_fp_neb_pathway"])
# dft_static_on_fp_neb claims to be for 999001|1 but its fp_structure is actually pathway 999002's --
# a structurally-plausible but wrong-pathway substitution (same atom count/species, different site).
wrong_pathway_struct = fp17["models"]["MyFP"]["full_fp_neb"]["pathways"]["999002|1"]["final_fp_neb_images"]["1"]["structure"]
fp17["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["fp_structure"] = wrong_pathway_struct
report17 = na.validate_neb_analysis_inputs(ref17, fp17, fp_order=["MyFP"])
check("17. dft_static_on_fp_neb fp_structure taken from a different pathway's image: "
      "validation catches the energy/force duplicated-value mismatch this causes",
      not report17.ok)

# ── 18. mismatched species order (Section I) ────────────────────────────────
ref18, fp18, _ = _base_dataset()
diag_sites = fp18["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["fp_structure"]["sites"]
diag_sites[0], diag_sites[1] = diag_sites[1], diag_sites[0]  # swap Na/Cl site order
report18 = na.validate_neb_analysis_inputs(ref18, fp18, fp_order=["MyFP"])
check("18. dft_static_on_fp_neb species order swapped vs full_fp_neb: validation catches it "
      "(no StructureMatcher-style reordering tolerance)", not report18.ok)

# ── 19. mismatched atom count (Section I) ───────────────────────────────────
ref19, fp19, _ = _base_dataset()
diag_sites19 = fp19["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["fp_structure"]["sites"]
diag_sites19.append(copy.deepcopy(diag_sites19[0]))
report19 = na.validate_neb_analysis_inputs(ref19, fp19, fp_order=["MyFP"])
check("19. dft_static_on_fp_neb atom count differs from full_fp_neb's: validation catches it", not report19.ok)

# ── 20. mismatched duplicated FP energy/force values (Section I) ───────────
ref20, fp20, _ = _base_dataset()
fp20["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["fp_energy_total_eV"] = -1.0
report20 = na.validate_neb_analysis_inputs(ref20, fp20, fp_order=["MyFP"])
check("20. dft_static_on_fp_neb duplicated fp_energy_total_eV disagrees with full_fp_neb's own value "
      "by far more than float noise: validation catches it", not report20.ok)

# ── 21. sparse but valid dft_static_on_fp_neb coverage (Section I) ─────────
ref21, fp21, ex21 = _base_dataset()
ref21["pathways"]["999002|1"] = copy.deepcopy(ex21["dft_reference_pathway"])
ref21["pathways"]["999002|1"]["identifiers"]["icsd_id"] = "999002"
ref21["common_pathway_keys"].append("999002|1")
fp21["models"]["MyFP"]["full_fp_neb"]["pathways"]["999002|1"] = copy.deepcopy(ex21["full_fp_neb_pathway"])
fp21["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999002|1"] = copy.deepcopy(ex21["fp_static_on_dft_neb_pathway"])
# dft_static_on_fp_neb deliberately has NO record at all for 999002|1 -- sparse coverage is allowed.
report21 = na.validate_neb_analysis_inputs(ref21, fp21, fp_order=["MyFP"])
check("21. dft_static_on_fp_neb sparse coverage (present for 999001|1, absent for 999002|1) is valid, "
      "not flagged as an error", report21.ok)

# ── 22. explicit missing expected pathway (Section G) ──────────────────────
ref22, fp22, _ = _base_dataset()
ref22["common_pathway_keys"].append("999003|1")  # listed as expected, but no pathway record exists for it
analysis22 = na.build_neb_analysis_results(ref22, fp22, validate=True)
check("22. pathway listed in common_pathway_keys but absent from pathways: "
      "does not crash build_neb_analysis_results, remains traceable via common_pathway_keys",
      "999003|1" in ref22["common_pathway_keys"] and "999003|1" not in ref22["pathways"])

# ── 23. unavailable protocol branch (Section G) ─────────────────────────────
ref23, fp23, ex23 = _base_dataset()
fp23_no_diag = {
    "schema_version": "1.0.0", "component": "ion_migration_neb", "dataset_name": "no-diagnostic-protocol",
    "units": {"energy": "eV", "forces": "eV/angstrom"},
    "models": {"MyFP": {
        "metadata": {"fp_key": "MyFP"},
        "full_fp_neb": {"pathways": {"999001|1": ex23["full_fp_neb_pathway"]}},
        "fp_static_on_dft_neb": {"pathways": {"999001|1": ex23["fp_static_on_dft_neb_pathway"]}},
        "dft_static_on_fp_neb": {"pathways": {}},  # protocol entirely unavailable for this FP
    }},
}
analysis23 = na.build_neb_analysis_results(ref23, fp23_no_diag, validate=True)
check("23. dft_static_on_fp_neb entirely unavailable: full_fp_neb_path_force_errors_df is None, "
      "not fabricated or substituted from another protocol",
      analysis23.full_fp_neb_path_force_errors_df is None)
check("23. the other two protocols still work normally when only one is unavailable",
      analysis23.dft_neb_path_force_errors_df is not None and len(analysis23.dft_path_metrics_df) == 1)

# ── 24. zero valid force-angle population, no RuntimeWarning (Section K) ───
import warnings
ref24, fp24, _ = _base_dataset()
zero_forces_all = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
fp24["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["fp_forces_eV_per_angstrom"] = zero_forces_all
fp24["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]["images"]["0"]["dft_forces_eV_per_angstrom"] = zero_forces_all
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    row = na._force_error_row(
        "MyFP", "999001", "1", 0, "initial", -10.0, zero_forces_all, -10.0, zero_forces_all,
        "dft_static_on_fp_neb", "test", "test", "example", "present")
check("24. all-zero-force image: mean/max force-angle error are NaN, n_valid_force_angles == 0, "
      "no RuntimeWarning raised",
      row["n_valid_force_angles"] == 0 and row["mean_force_angle_error_deg"] != row["mean_force_angle_error_deg"]
      and row["max_force_angle_error_deg"] != row["max_force_angle_error_deg"])  # NaN != NaN

# ── 25. external example running with warnings treated as errors (Section K) ──
with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        ex25 = na.example_canonical_pathway_records()
        ref25 = {
            "schema_version": "1.0.0", "component": "ion_migration_neb", "dataset_name": "warnings-as-errors example",
            "units": {"energy": "eV", "forces": "eV/angstrom"},
            "common_pathway_keys": ["999001|1"],
            "pathways": {"999001|1": ex25["dft_reference_pathway"]},
        }
        fp25 = {
            "schema_version": "1.0.0", "component": "ion_migration_neb", "dataset_name": "warnings-as-errors example",
            "units": {"energy": "eV", "forces": "eV/angstrom"},
            "models": {"MyFP": {
                "metadata": {"fp_key": "MyFP"},
                "full_fp_neb": {"pathways": {"999001|1": ex25["full_fp_neb_pathway"]}},
                "fp_static_on_dft_neb": {"pathways": {"999001|1": ex25["fp_static_on_dft_neb_pathway"]}},
                "dft_static_on_fp_neb": {"pathways": {"999001|1": ex25["dft_static_on_fp_neb_pathway"]}},
            }},
        }
        na.build_neb_analysis_results(ref25, fp25, validate=True)
        check("25. example_canonical_pathway_records() + build_neb_analysis_results run "
              "with warnings-as-errors: no warning raised", True)
    except Warning as w:
        check(f"25. warnings-as-errors run raised a warning: {w}", False)

# ── 26. external dataset with a pathway count other than 155/154 ───────────
ref26, fp26, _ = _base_dataset()
report26 = na.validate_neb_analysis_inputs(ref26, fp26, expected_pathways=3, fp_order=["MyFP"])
check("26. external 1-pathway dataset checked against expected_pathways=3: mismatch correctly flagged, "
      "no assumption that 155/154 is the only valid count",
      not report26.ok and any("== 3" in c["check"] for c in report26.checks))
report26b = na.validate_neb_analysis_inputs(ref26, fp26, expected_pathways=None, fp_order=["MyFP"])
check("26b. no expected_pathways given: a 1-pathway dataset validates fine on every other check",
      all(c["ok"] for c in report26b.checks if not c["check"].startswith("DFT reference pathway count")))

# ── 27-30. unsuccessful_pathways branch: failed / missing / interrupted / ──
# ── not-run full_fp_neb records are preserved, validated, reported in     ──
# ── coverage, and never enter any scientific metric population           ──
for status27 in ("failed", "missing", "interrupted", "not_run"):
    ref27, fp27, ex27 = _base_dataset()
    ref27["pathways"]["999002|1"] = copy.deepcopy(ex27["dft_reference_pathway"])
    ref27["pathways"]["999002|1"]["identifiers"]["icsd_id"] = "999002"
    ref27["common_pathway_keys"].append("999002|1")
    fp27["models"]["MyFP"]["full_fp_neb"]["unsuccessful_pathways"] = {
        "999002|1": {
            "identifiers": {"icsd_id": "999002", "source_path_id": "1"},
            "calculation_status": status27,
            "error": None if status27 in ("missing", "not_run") else f"example {status27} error",
        }
    }
    report27 = na.validate_neb_analysis_inputs(ref27, fp27, fp_order=["MyFP"])
    check(f"27. unsuccessful_pathways calculation_status={status27!r}: validation report stays ok "
          "(well-formed unsuccessful record)", report27.ok)

    unsuccessful27 = na.collect_unsuccessful_records(fp27, ["MyFP"])
    check(f"27. collect_unsuccessful_records({status27!r}) surfaces the record under full_fp_neb",
          unsuccessful27["MyFP"]["full_fp_neb"].get("999002|1", {}).get("calculation_status") == status27)

    coverage27 = na.build_protocol_coverage_table(ref27, fp27, ["MyFP"])
    row27 = coverage27[(coverage27["fp_key"] == "MyFP") & (coverage27["protocol"] == "full_fp_neb")].iloc[0]
    check(f"27. build_protocol_coverage_table({status27!r}) reports unsuccessful_count=1 and the status in "
          "unsuccessful_status_counts",
          row27["unsuccessful_count"] == 1 and row27["unsuccessful_status_counts"].get(status27) == 1)

    analysis27 = na.build_neb_analysis_results(ref27, fp27, validate=True)
    check(f"27. unsuccessful_pathways record ({status27!r}) never enters dft_path_metrics_df/"
          "fp_path_metrics_by_protocol (pathway 999002|1 absent from full_fp_neb's own metrics)",
          "999002" not in set(analysis27.fp_path_metrics_by_protocol["MyFP"]["full_fp_neb"]["ICSD"].astype(str)))
    check(f"27. unsuccessful_pathways record ({status27!r}) surfaced on the returned object via "
          "unsuccessful_records_by_fp_protocol",
          analysis27.unsuccessful_records_by_fp_protocol["MyFP"]["full_fp_neb"]["999002|1"]["calculation_status"]
          == status27)

# ── 28. unsuccessful_pathways with an invalid calculation_status value is flagged ──
ref28, fp28, _ = _base_dataset()
fp28["models"]["MyFP"]["full_fp_neb"]["unsuccessful_pathways"] = {
    "999001|9": {
        "identifiers": {"icsd_id": "999001", "source_path_id": "9"},
        "calculation_status": "completed",   # invalid here: "completed" records belong in "pathways", not here
        "error": None,
    }
}
report28 = na.validate_neb_analysis_inputs(ref28, fp28, fp_order=["MyFP"])
check("28. unsuccessful_pathways record with calculation_status='completed' (should never appear here): "
      "validation catches it", not report28.ok)

# ── 29. unsuccessful_image_attempts (dft_static_on_fp_neb): identifiers, status ──
ref29, fp29, _ = _base_dataset()
fp29["models"]["MyFP"]["dft_static_on_fp_neb"]["unsuccessful_image_attempts"] = {
    "MyFP|999001|1|1": {
        "fp_key": "MyFP", "pathway_key": "999001|1", "image_index": 1,
        "status": "failed", "error": "example VASP parse failure",
        "calculation_provenance": "test",
    }
}
report29 = na.validate_neb_analysis_inputs(ref29, fp29, fp_order=["MyFP"])
check("29. unsuccessful_image_attempts record with consistent identifiers and a valid status: "
      "validation report stays ok", report29.ok)
unsuccessful29 = na.collect_unsuccessful_records(fp29, ["MyFP"])
check("29. collect_unsuccessful_records surfaces the dft_static_on_fp_neb unsuccessful_image_attempts record",
      unsuccessful29["MyFP"]["dft_static_on_fp_neb"]["MyFP|999001|1|1"]["status"] == "failed")

# ── 30. unsuccessful_image_attempts with a key/identifier mismatch is flagged ──
ref30, fp30, _ = _base_dataset()
fp30["models"]["MyFP"]["dft_static_on_fp_neb"]["unsuccessful_image_attempts"] = {
    "MyFP|999001|1|1": {
        "fp_key": "MyFP", "pathway_key": "999001|1", "image_index": 2,   # mismatched: key says image 1
        "status": "failed", "error": "example",
        "calculation_provenance": "test",
    }
}
report30 = na.validate_neb_analysis_inputs(ref30, fp30, fp_order=["MyFP"])
check("30. unsuccessful_image_attempts record whose key does not match its own image_index: "
      "validation catches it", not report30.ok)

# ── 31. partial fp_static_on_dft_neb evaluation (some but not all images present) ──
# is caught pre-flight by validate_neb_analysis_inputs's index-set check, and
# build_neb_analysis_results itself refuses to zip()-pair the mismatched images
# (raises, never silently truncates) -- exactly the existing zip()-pairing
# safety behavior test 6 already exercises for a different malformed field.
ref31, fp31, _ = _base_dataset()
del fp31["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]["images"]["1"]
report31 = na.validate_neb_analysis_inputs(ref31, fp31, fp_order=["MyFP"])
check("31. partial fp_static_on_dft_neb (image 1 missing): validation catches it via the index-set check",
      not report31.ok)
try:
    na.build_neb_analysis_results(ref31, fp31, validate=False)
    check("31. partial fp_static_on_dft_neb: build_neb_analysis_results refuses to zip()-pair "
          "mismatched images", False)
except ValueError:
    check("31. partial fp_static_on_dft_neb: build_neb_analysis_results raises ValueError instead of "
          "silently zip()-pairing mismatched images", True)

# ── 31b. the same coverage check, run directly (validate_analysis_coverage), ──
# confirms coverage_mismatch_df's own reporting shape on already-built image maps.
ref31b, fp31b, _ = _base_dataset()
dft_images_31b, _ = na.build_dft_neb_image_map(ref31b)
fp_static_31b = {"MyFP": {"999001|1": dft_images_31b["999001|1"][:2]}}   # 2 images, not 3
coverage_mismatch_31b = na.validate_analysis_coverage(fp_static_31b, dft_images_31b, ["MyFP"])
check("31b. validate_analysis_coverage reports a count mismatch for a genuinely short image list",
      len(coverage_mismatch_31b) == 1 and coverage_mismatch_31b.iloc[0]["issue"] == "image count mismatch")

# ── 32. new-schema-only neb_status (no "convergence" key; the current ──────
# generator's real field names last_fmax_eV_per_angstrom/optimizer_steps
# instead of the legacy neb_last_fmax/neb_n_steps) -- the exact schema
# fp_neb_generation_and_run.ipynb's merge_full_fp_neb actually produces,
# confirmed against a real Zaratan mace_matpes_pbe run. Regression test for
# a real KeyError('neb_last_fmax') this schema previously triggered in
# build_full_fp_neb_status_map.
ref32, fp32, _ = _base_dataset()
pdata32 = fp32["models"]["MyFP"]["full_fp_neb"]["pathways"]["999001|1"]
pdata32.pop("convergence", None)
pdata32["neb_status"] = {
    "neb_converged": True,
    "last_fmax_eV_per_angstrom": 0.031,
    "optimizer_steps": 17,
}
status_map32 = na.build_full_fp_neb_status_map(fp32, ["MyFP"])
rec32 = status_map32["MyFP"][("999001", "1")]
check("32. new-schema-only neb_status (no convergence key, last_fmax_eV_per_angstrom/"
      "optimizer_steps field names): build_full_fp_neb_status_map succeeds without KeyError "
      "and reports the real values",
      rec32.get("neb_converged") is True
      and abs(rec32.get("neb_last_fmax") - 0.031) < 1e-9
      and rec32.get("neb_n_steps") == 17)

# ── 32b. neither the legacy nor the new field name present at all -- must ──
# raise clearly, never silently default to None/0 and misreport a real
# number as missing.
ref32b, fp32b, _ = _base_dataset()
pdata32b = fp32b["models"]["MyFP"]["full_fp_neb"]["pathways"]["999001|1"]
pdata32b.pop("convergence", None)
pdata32b["neb_status"] = {"neb_converged": True}  # no fmax/steps field under any known name
try:
    na.build_full_fp_neb_status_map(fp32b, ["MyFP"])
    check("32b. neither legacy nor new fmax/steps field name present: build_full_fp_neb_status_map "
          "raises instead of silently defaulting", False)
except KeyError:
    check("32b. neither legacy nor new fmax/steps field name present: build_full_fp_neb_status_map "
          "raises instead of silently defaulting", True)

# ── 33. genuinely empty dft_static_on_fp_neb / fp_static_on_dft_neb records ──
# (a normal, expected state early in any real generation run -- e.g. the
# VASP-diagnostics protocol not run yet for any FP) must produce a
# correctly-columned, 0-row DataFrame from build_full_fp_neb_path_force_errors
# / build_dft_neb_path_force_errors, not pandas' default 0-row/0-column
# DataFrame. Regression test for a real KeyError('fp_key') this triggered in
# neb_analysis.ipynb cell 47's
# `.groupby(['fp_key','icsd_id','source_path_id']).ngroups` when the user fed
# a genuinely partial real dataset through it.
empty_force_errors_df = na.build_full_fp_neb_path_force_errors({}, ["MyFP", "OtherFP"])
check("33. build_full_fp_neb_path_force_errors on genuinely empty input returns 0 rows",
      len(empty_force_errors_df) == 0)
check("33. ...with the real columns present (not pandas' default 0-row/0-column empty frame)",
      list(empty_force_errors_df.columns) == na._FORCE_ERROR_ROW_COLUMNS)
try:
    n_groups33 = empty_force_errors_df.groupby(["fp_key", "icsd_id", "source_path_id"]).ngroups
    check("33. .groupby(['fp_key','icsd_id','source_path_id']).ngroups on the empty result "
          "succeeds (exactly cell 47's real crash site) and reports 0 groups",
          n_groups33 == 0)
except KeyError:
    check("33. .groupby(['fp_key','icsd_id','source_path_id']).ngroups on the empty result "
          "succeeds (exactly cell 47's real crash site) and reports 0 groups", False)

empty_dft_neb_force_errors_df = na.build_dft_neb_path_force_errors({}, {}, ["MyFP"])
check("33b. build_dft_neb_path_force_errors on genuinely empty input also returns 0 rows "
      "with the real columns present",
      len(empty_dft_neb_force_errors_df) == 0
      and list(empty_dft_neb_force_errors_df.columns) == na._FORCE_ERROR_ROW_COLUMNS)

# Non-empty inputs must be completely unaffected by the empty-case fix.
# build_full_fp_neb_path_force_errors expects protocol dft_static_on_fp_neb's
# shape (DFT single-point on the FP's own final images: both dft_* and fp_*
# energy/forces per image), not fp_static_on_dft_neb's.
ref33c, fp33c, _ = _base_dataset()
dft_static_records_33c = {"MyFP": {"999001|1": fp33c["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]}}
nonempty_df33c = na.build_full_fp_neb_path_force_errors(dft_static_records_33c, ["MyFP"])
check("33c. a genuinely non-empty result is unaffected by the empty-case fix (real rows, real columns)",
      len(nonempty_df33c) > 0 and list(nonempty_df33c.columns) == na._FORCE_ERROR_ROW_COLUMNS)

# ── 34. barrier_combination: "pooled" (default) vs "round_then_average_ ────
# legacy" -- two synthetic full_fp_neb pathways under one FP, hand-chosen so
# the forward barrier errors are [0.101, 0.103] and the backward barrier
# errors are [0.201, 0.207] (both DFT profiles: E0=-10.000, Emid=-9.000,
# E2=-9.500, a Normal-Hill profile with asymmetric endpoints so forward and
# backward barriers are independently controllable via the FP endpoint/
# intermediate energies). This is a real regression pin, not a tautological
# call-and-assert: every target string below was computed independently by
# hand (see the comment on each check).
import pandas as pd
from neb_plots import area_between_curves, simplify_class

ref34, fp34, ex34 = _base_dataset()


def _make_pathway_pair(icsd, path_id, dft_e0, dft_emid, dft_e2, fp_e0, fp_emid, fp_e2):
    dft_p = copy.deepcopy(ex34["dft_reference_pathway"])
    dft_p["identifiers"]["icsd_id"] = icsd
    dft_p["identifiers"]["source_path_id"] = path_id
    dft_p["dft_neb_images"][0]["energy_total_eV"] = dft_e0
    dft_p["dft_neb_images"][1]["energy_total_eV"] = dft_emid
    dft_p["dft_neb_images"][2]["energy_total_eV"] = dft_e2
    dft_p["dft_relaxed_endpoints"]["initial"]["energy_total_eV"] = dft_e0
    dft_p["dft_relaxed_endpoints"]["final"]["energy_total_eV"] = dft_e2
    fp_p = copy.deepcopy(ex34["full_fp_neb_pathway"])
    fp_p["identifiers"]["icsd_id"] = icsd
    fp_p["identifiers"]["source_path_id"] = path_id
    fp_p["final_fp_neb_images"]["0"]["fp_energy_total_eV"] = fp_e0
    fp_p["final_fp_neb_images"]["1"]["fp_energy_total_eV"] = fp_emid
    fp_p["final_fp_neb_images"]["2"]["fp_energy_total_eV"] = fp_e2
    return dft_p, fp_p


# Path A: DFT forward=1.000, backward=0.500; FP forward=1.101 (err +0.101),
# backward=0.701 (err +0.201).
dft_a34, fp_a34 = _make_pathway_pair("999101", "1", -10.000, -9.000, -9.500,
                                      -10.000, -8.899, -9.600)
# Path B: same DFT profile; FP forward=1.103 (err +0.103), backward=0.707
# (err +0.207).
dft_b34, fp_b34 = _make_pathway_pair("999103", "1", -10.000, -9.000, -9.500,
                                      -10.000, -8.897, -9.604)

ref34["pathways"] = {"999101|1": dft_a34, "999103|1": dft_b34}
ref34["common_pathway_keys"] = ["999101|1", "999103|1"]
fp34["models"]["MyFP"]["full_fp_neb"]["pathways"] = {"999101|1": fp_a34, "999103|1": fp_b34}
# fp_static_on_dft_neb/dft_static_on_fp_neb: this test only exercises the
# "full" protocol's barrier_combination behavior, so the static-protocol
# content itself doesn't matter -- but it must be non-empty (real converged
# data for both pathway keys), not {}, since compute_barrier_error_summaries'
# static_barrier_df must have real MAE_energy_forward_barrier/etc. columns
# to build at all (a genuinely empty-protocol DataFrame is a distinct,
# already-covered case -- see test 23 -- and not what this test is about).
for pkey in ("999101|1", "999103|1"):
    static_p = copy.deepcopy(ex34["fp_static_on_dft_neb_pathway"])
    icsd, path_id = pkey.split("|")
    static_p["identifiers"] = {"icsd_id": icsd, "source_path_id": path_id}
    fp34["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"][pkey] = static_p
    dft_static_p = copy.deepcopy(ex34["dft_static_on_fp_neb_pathway"])
    dft_static_p["identifiers"] = {"icsd_id": icsd, "source_path_id": path_id}
    fp34["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"][pkey] = dft_static_p
del fp34["models"]["MyFP"]["fp_static_on_dft_neb"]["pathways"]["999001|1"]
del fp34["models"]["MyFP"]["dft_static_on_fp_neb"]["pathways"]["999001|1"]

analysis34 = na.build_neb_analysis_results(ref34, fp34, validate=True)
benchmark_pathways_df34 = pd.DataFrame(
    [{"ICSD": int(i), "Path": int(p)} for i, p in (k.split("|") for k in ref34["common_pathway_keys"])]
)


def _summary34(**kwargs):
    return na.compute_key_neb_metrics_summary(
        analysis34.dft_valid_path_metrics_df, analysis34.fp_path_metrics_by_protocol,
        analysis34.dft_path_metrics_df, analysis34.full_fp_neb_status_by_fp_path,
        analysis34.endpoint_rmsd_by_fp_protocol_path, benchmark_pathways_df34,
        analysis34.dft_neb_images_by_path, analysis34.full_fp_neb_images_by_fp_path,
        analysis34.fp_static_on_dft_neb_images_by_fp_path,
        ["MyFP"], 1.0, area_between_curves, simplify_class, **kwargs,
    )


legacy34 = _summary34(barrier_combination="round_then_average_legacy")
pooled34 = _summary34(barrier_combination="pooled")
default34 = _summary34()

# By hand: MAE_fwd = mean(0.101, 0.103) = 0.102 -> round(.,2) = 0.10;
# MAE_bwd = mean(0.201, 0.207) = 0.204 -> round(.,2) = 0.20;
# combined MAE = (0.10 + 0.20) / 2 = 0.15. RMSE_fwd = sqrt(mean(0.101**2,
# 0.103**2)) = 0.10200... -> round(.,2) = 0.10; RMSE_bwd = sqrt(mean(0.201**2,
# 0.207**2)) = 0.20402... -> round(.,2) = 0.20; combined RMSE =
# sqrt((0.10**2 + 0.20**2) / 2) = 0.158113... -> "0.1581". This reproduces
# the exact real-world defect: two genuinely different underlying MAE values
# (0.102, 0.204) collapse onto the same displayed 0.15 a real published
# barrier value would show, because of the intermediate 2dp rounding.
check("34. round_then_average_legacy: forward/backward MAE (0.102, 0.204) round to (0.10, 0.20) "
      "BEFORE combining, giving 0.1500 / 0.1581 -- matches independently hand-computed values",
      legacy34.loc["MyFP", "Barrier error (eV) (full)"] == "0.1500 / 0.1581")

# By hand, pooled: MAE over the concatenated [0.101, 0.103, 0.201, 0.207]
# array = 0.612 / 4 = 0.153; RMSE = sqrt(mean([0.101**2, 0.103**2, 0.201**2,
# 0.207**2])) = sqrt(0.10406 / 4) = 0.161291... -> "0.1613". Genuinely
# different from the legacy 0.1500 / 0.1581 above precisely because no
# intermediate rounding occurs.
check("34. pooled: MAE/RMSE over the concatenated forward+backward error array give "
      "0.1530 / 0.1613 -- independently hand-computed, and different from the legacy value",
      pooled34.loc["MyFP", "Barrier error (eV) (full)"] == "0.1530 / 0.1613")

check("34. barrier_combination default (no argument passed) matches the explicit 'pooled' call",
      default34.loc["MyFP", "Barrier error (eV) (full)"] == pooled34.loc["MyFP", "Barrier error (eV) (full)"])

try:
    _summary34(barrier_combination="bogus_value")
    check("34. an invalid barrier_combination value raises ValueError instead of silently "
          "falling back to one convention", False)
except ValueError:
    check("34. an invalid barrier_combination value raises ValueError instead of silently "
          "falling back to one convention", True)

# ── 35. format_half_away_from_zero: the project's display-time rounding ────
# house rule (round halves AWAY FROM ZERO, contrasted directly against
# Python/NumPy's default round-half-to-even) established alongside this
# fix. 0.125 is exactly representable in binary, so this is a genuine tie,
# not float noise -- Python's own round(0.125, 2) gives 0.12 (rounds to
# even), proving the two conventions really do disagree here.
check("35. format_half_away_from_zero(0.125, 2) rounds the tie up to '0.13'",
      na.format_half_away_from_zero(0.125, 2) == "0.13")
check("35. Python's own round(0.125, 2) gives 0.12 (round-half-to-even) -- confirms 0.125 is "
      "a genuine tie where the two conventions actually disagree, not a coincidence",
      round(0.125, 2) == 0.12)
check("35. format_half_away_from_zero(-0.125, 2) rounds the tie to '-0.13' (away from zero, "
      "not toward positive infinity)",
      na.format_half_away_from_zero(-0.125, 2) == "-0.13")
check("35. format_half_away_from_zero(0.135, 2) rounds the tie up to '0.14' (0.135 is not "
      "exactly representable in binary, but Decimal(repr(x)) recovers the intended decimal "
      "tie rather than the binary value's true side)",
      na.format_half_away_from_zero(0.135, 2) == "0.14")
check("35. format_half_away_from_zero(0.0647, 4) is a no-op away from any tie: '0.0647'",
      na.format_half_away_from_zero(0.0647, 4) == "0.0647")
check("35. format_half_away_from_zero(None, 4) and format_half_away_from_zero(float('nan'), 4) "
      "both return 'nan'",
      na.format_half_away_from_zero(None, 4) == "nan"
      and na.format_half_away_from_zero(float("nan"), 4) == "nan")


# ── summary ──────────────────────────────────────────────────────────────
n_pass = sum(1 for _, ok in results if ok)
print(f"\n{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("FAILURES:")
    for name, ok in results:
        if not ok:
            print("  -", name)
    sys.exit(1)
sys.exit(0)
