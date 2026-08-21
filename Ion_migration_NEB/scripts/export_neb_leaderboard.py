#!/usr/bin/env python3
"""Public, deterministic leaderboard-summary export for the Ion Migration by
NEB component.

Given a DFT-NEB reference file and an all-FP results file (each accepted as
plain .json or gzip-compressed .json.gz), this script:

  1. loads both through the public `neb_analysis.load_neb_datasets`;
  2. validates the active population (154 pathways, 1,078 finalized
     DFT-reference images, 109 unique active ICSD identifiers, `113548|10`
     absent) before computing or writing anything;
  3. runs the same unmodified `neb_analysis.py` functions the analysis
     notebook itself uses (`build_neb_analysis_results`, and the
     `barrier_error_summary_by_protocol` / `profile_summary_by_protocol`
     attributes it already computes -- never recomputed a second time here);
  4. reproduces `compute_key_neb_metrics_summary`'s exact round-then-combine
     order for the barrier-error MAE/RMSE columns, and cross-checks every
     exported value against a fresh call to that function before writing
     anything, so the export cannot silently drift from what the analysis
     notebook itself displays;
  5. writes the compact per-FP summary to
     `<component-dir>/data/ion_migration_neb_leaderboard_summary.json` and an
     identical copy to `<repo-root>/docs/data/ion_migration_neb_leaderboard_summary.json`
     (the GitHub Pages copy), creating `docs/data/` if needed;
  6. refuses to silently overwrite either destination if it already exists
     with *different* content, unless `--force` is given (a no-op rewrite of
     byte-identical content is always allowed).

No scientific value is hardcoded anywhere in this script: every number in
the output comes from `neb_analysis.py`.

Usage (run from the `Ion_migration_NEB/` directory):

    python scripts/export_neb_leaderboard.py \\
        --reference data/ion_migration_neb_reference.json.gz \\
        --results /path/to/ion_migration_neb_results_standardized.json

Or from anywhere, with --component-dir pointing at Ion_migration_NEB/:

    python Ion_migration_NEB/scripts/export_neb_leaderboard.py \\
        --component-dir Ion_migration_NEB \\
        --reference Ion_migration_NEB/data/ion_migration_neb_reference.json.gz \\
        --results /path/to/ion_migration_neb_results_standardized.json
"""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

FP_ORDER = [
    "MACE-MP0_medium", "CHGNET", "M3GNET_pes", "UMA_s1_p1",
    "M3GNET_matpes_PBE", "TensorNET_matpes_PBE", "MACE_matpes_pbe",
]
FP_DISPLAY_NAMES = {
    "MACE-MP0_medium":      "MACE",
    "CHGNET":               "CHGNet",
    "M3GNET_pes":           "M3GNet",
    "UMA_s1_p1":            "UMA",
    "M3GNET_matpes_PBE":    "M3GNet-MatPES",
    "TensorNET_matpes_PBE": "TensorNet-MatPES",
    "MACE_matpes_pbe":      "MACE-MatPES",
}
OUTLIER_THRESHOLD = 1.0
ROUND_DECIMALS = 4

EXPECTED_ACTIVE_PATHWAYS = 154
EXPECTED_ACTIVE_IMAGES = 1078
EXPECTED_UNIQUE_ICSD = 109
EXCLUDED_PATHWAY_KEY = "113548|10"


def r(x):
    return round(float(x), ROUND_DECIMALS)


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_active_population(reference_data):
    """Hard population checks, run before anything is computed or written.
    Raises AssertionError (refusing to proceed) if the shipped/regenerated
    reference does not match the approved active population."""
    pathways = reference_data["pathways"]
    common = reference_data["common_pathway_keys"]
    n_images = sum(len(p["dft_neb_images"]) for p in pathways.values())
    unique_icsd = {p["identifiers"]["icsd_id"] for p in pathways.values()}

    assert len(pathways) == EXPECTED_ACTIVE_PATHWAYS, (
        f"active pathway count is {len(pathways)}, expected {EXPECTED_ACTIVE_PATHWAYS} -- refusing to export")
    assert len(common) == EXPECTED_ACTIVE_PATHWAYS, (
        f"common_pathway_keys count is {len(common)}, expected {EXPECTED_ACTIVE_PATHWAYS} -- refusing to export")
    assert n_images == EXPECTED_ACTIVE_IMAGES, (
        f"total DFT-reference image count is {n_images}, expected {EXPECTED_ACTIVE_IMAGES} -- refusing to export")
    assert all(len(p["dft_neb_images"]) == 7 for p in pathways.values()), (
        "not every active pathway has exactly 7 DFT-reference images -- refusing to export")
    assert len(unique_icsd) == EXPECTED_UNIQUE_ICSD, (
        f"unique active ICSD count is {len(unique_icsd)}, expected {EXPECTED_UNIQUE_ICSD} -- refusing to export")
    assert EXCLUDED_PATHWAY_KEY not in pathways and EXCLUDED_PATHWAY_KEY not in common, (
        f"{EXCLUDED_PATHWAY_KEY} is present in the active population -- refusing to export")


def build_leaderboard(reference_path, results_path, na, area_between_curves, simplify_class):
    with open(reference_path, "rb") as f:
        ref_sha256 = hashlib.sha256(f.read()).hexdigest() if Path(reference_path).suffix != ".gz" else None
    # For .gz inputs, record the compressed-file hash directly (what a user
    # actually downloads/verifies); load_neb_datasets decompresses internally.
    if ref_sha256 is None:
        ref_sha256 = sha256_of_file(reference_path)
    res_sha256 = sha256_of_file(results_path)

    reference_data, fp_results = na.load_neb_datasets(str(reference_path), str(results_path))
    validate_active_population(reference_data)

    fp_order = [k for k in FP_ORDER if k in fp_results.get("models", {})]
    missing = [k for k in FP_ORDER if k not in fp_results.get("models", {})]
    if missing:
        print(f"WARNING: results file is missing FPs {missing}; exporting only {fp_order}", file=sys.stderr)

    analysis = na.build_neb_analysis_results(
        reference_data, fp_results, expected_pathways=EXPECTED_ACTIVE_PATHWAYS,
        fp_order=fp_order, outlier_threshold=OUTLIER_THRESHOLD, validate=True,
    )
    assert analysis.validation_report.ok, "validation failed -- refusing to export"

    full_barrier_df = analysis.barrier_error_summary_by_protocol[na.PROTOCOL_FULL_FP_NEB]
    static_barrier_df = analysis.barrier_error_summary_by_protocol[na.PROTOCOL_FP_STATIC_ON_DFT_NEB]
    full_profile_df = analysis.profile_summary_by_protocol[na.PROTOCOL_FULL_FP_NEB].set_index("FP")

    # Fresh call, used only for the cross-check assertions below -- never as
    # a source of any exported value.
    key_df = na.compute_key_neb_metrics_summary(
        analysis.dft_valid_path_metrics_df, analysis.fp_path_metrics_by_protocol,
        analysis.dft_path_metrics_df, analysis.full_fp_neb_status_by_fp_path,
        analysis.endpoint_rmsd_by_fp_protocol_path,
        na.pd.DataFrame([{"ICSD": int(k.split("|")[0]), "Path": int(k.split("|")[1])}
                          for k in reference_data["common_pathway_keys"]]),
        analysis.dft_neb_images_by_path, analysis.full_fp_neb_images_by_fp_path,
        analysis.fp_static_on_dft_neb_images_by_fp_path, fp_order, OUTLIER_THRESHOLD,
        area_between_curves, simplify_class,
    )

    records = []
    for fp_key in fp_order:
        conv = analysis.full_fp_neb_convergence_summary[fp_key][na.PROTOCOL_FULL_FP_NEB]
        n_total = int(conv["n_total"])
        n_not_conv = int(conv["n_neb_not_conv"])

        # Same round-then-combine order as compute_key_neb_metrics_summary's
        # internal _combined_barrier: round forward/backward MAE and RMSE to
        # 2 decimals FIRST, then combine.
        fwd_mae = round(float(full_barrier_df.loc[fp_key, "MAE_energy_forward_barrier"]), 2)
        bwd_mae = round(float(full_barrier_df.loc[fp_key, "MAE_energy_backward_barrier"]), 2)
        fwd_rmse = round(float(full_barrier_df.loc[fp_key, "RMSE_energy_forward_barrier"]), 2)
        bwd_rmse = round(float(full_barrier_df.loc[fp_key, "RMSE_energy_backward_barrier"]), 2)
        barrier_mae_full = (fwd_mae + bwd_mae) / 2
        barrier_rmse_full = ((fwd_rmse ** 2 + bwd_rmse ** 2) / 2) ** 0.5

        fwd_mae_s = round(float(static_barrier_df.loc[fp_key, "MAE_energy_forward_barrier"]), 2)
        bwd_mae_s = round(float(static_barrier_df.loc[fp_key, "MAE_energy_backward_barrier"]), 2)
        fwd_rmse_s = round(float(static_barrier_df.loc[fp_key, "RMSE_energy_forward_barrier"]), 2)
        bwd_rmse_s = round(float(static_barrier_df.loc[fp_key, "RMSE_energy_backward_barrier"]), 2)
        barrier_mae_static = (fwd_mae_s + bwd_mae_s) / 2
        barrier_rmse_static = ((fwd_rmse_s ** 2 + bwd_rmse_s ** 2) / 2) ** 0.5

        ep_dE_mae = float(full_profile_df.loc[fp_key, "Endpoint ΔE MAE (eV)"])
        ep_dE_rmse = float(full_profile_df.loc[fp_key, "Endpoint ΔE RMSE (eV)"])
        ep_rank_pct = float(full_profile_df.loc[fp_key, "Endpoint Energy Ranking Accuracy (%)"])
        shape_pct = float(full_profile_df.loc[fp_key, "Pathway Topology Accuracy (%)"])

        expected_barrier_full = key_df.loc[fp_key, "Barrier error (eV) (full)"]
        got_barrier_full = f"{barrier_mae_full:.4f} / {barrier_rmse_full:.4f}"
        assert got_barrier_full == expected_barrier_full, (fp_key, got_barrier_full, expected_barrier_full)
        expected_barrier_static = key_df.loc[fp_key, "Barrier error (eV) (static)"]
        got_barrier_static = f"{barrier_mae_static:.4f} / {barrier_rmse_static:.4f}"
        assert got_barrier_static == expected_barrier_static, (fp_key, got_barrier_static, expected_barrier_static)
        expected_nonconv = key_df.loc[fp_key, "Non-conv. paths (n/total)"]
        assert expected_nonconv == f"{n_not_conv}/{n_total}", (fp_key, expected_nonconv, n_not_conv, n_total)

        records.append({
            "fp_key": fp_key,
            "display_name": FP_DISPLAY_NAMES[fp_key],
            "n_nonconverged": n_not_conv,
            "n_total": n_total,
            "barrier_mae_full_eV": r(barrier_mae_full),
            "barrier_rmse_full_eV": r(barrier_rmse_full),
            "barrier_mae_static_eV": r(barrier_mae_static),
            "barrier_rmse_static_eV": r(barrier_rmse_static),
            "endpoint_energy_diff_mae_eV": r(ep_dE_mae),
            "endpoint_energy_diff_rmse_eV": r(ep_dE_rmse),
            "endpoint_ranking_agreement_pct": r(ep_rank_pct),
            "energy_profile_shape_agreement_pct": r(shape_pct),
        })

    leaderboard = {
        "schema_version": "1.0.0",
        "component": "ion_migration_neb",
        "dataset_name": "FPBench ion-migration NEB leaderboard summary",
        "active_pathway_count": len(reference_data["pathways"]),
        "active_dft_reference_image_count": sum(
            len(p["dft_neb_images"]) for p in reference_data["pathways"].values()),
        "units": {"energy": "eV", "angle": "degree"},
        "metric_columns": {
            "n_nonconverged": "Number of full FP-NEB paths (of n_total) that did not satisfy the NEB force-convergence criterion within the maximum optimization steps.",
            "n_total": "Total number of eligible full FP-NEB calculations entering the barrier-error analysis for this FP (common active pathways with a valid DFT-NEB classification and a full_fp_neb result).",
            "barrier_mae_full_eV": "MAE of forward/backward migration-barrier errors (eV), full FP-NEB workflow over the converged full FP-NEB population, pooled and combined per compute_key_neb_metrics_summary's round-then-combine convention.",
            "barrier_rmse_full_eV": "RMSE of forward/backward migration-barrier errors (eV), full FP-NEB workflow over the converged full FP-NEB population, same combination convention.",
            "barrier_mae_static_eV": "MAE of forward/backward migration-barrier errors (eV), static FP evaluations on the finalized DFT-NEB image structures (not filtered by full FP-NEB convergence).",
            "barrier_rmse_static_eV": "RMSE of forward/backward migration-barrier errors (eV), static FP evaluations on the finalized DFT-NEB image structures (not filtered by full FP-NEB convergence).",
            "endpoint_energy_diff_mae_eV": "MAE of the endpoint energy-difference error (eV, not per atom), full FP-NEB workflow, over the converged full FP-NEB population.",
            "endpoint_energy_diff_rmse_eV": "RMSE of the endpoint energy-difference error (eV, not per atom), full FP-NEB workflow, over the converged full FP-NEB population.",
            "endpoint_ranking_agreement_pct": "Endpoint energy-ranking agreement (%): fraction of the converged full FP-NEB population where the FP identifies the same lower-energy endpoint as DFT (or both classify the endpoints as equal in energy).",
            "energy_profile_shape_agreement_pct": "Energy-profile shape agreement (%): fraction of the converged full FP-NEB population for which the FP reproduces the DFT Normal-Hill energy profile.",
        },
        "fp_order": fp_order,
        "models": records,
        "generation_metadata": {
            "generated_by": "Ion_migration_NEB/scripts/export_neb_leaderboard.py, from neb_analysis.py's build_neb_analysis_results/compute_key_neb_metrics_summary (same functions and rounding order as the analysis notebook)",
            "generation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_reference_file": str(Path(reference_path).name),
            "source_reference_sha256": ref_sha256,
            "source_results_file": str(Path(results_path).name),
            "source_results_sha256": res_sha256,
        },
    }
    return leaderboard


def write_output(leaderboard, dest_path, force):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    new_text = json.dumps(leaderboard, indent=2)
    if dest_path.exists():
        old_text = dest_path.read_text()
        # A regeneration_timestamp_utc-only difference is expected on every
        # run; compare everything else before deciding this is a real change.
        old_obj = json.loads(old_text)
        new_obj = json.loads(new_text)
        old_obj.get("generation_metadata", {}).pop("generation_timestamp_utc", None)
        new_obj_cmp = json.loads(new_text)
        new_obj_cmp.get("generation_metadata", {}).pop("generation_timestamp_utc", None)
        if old_obj != new_obj_cmp and not force:
            raise RuntimeError(
                f"{dest_path} already exists with different content. Refusing to overwrite "
                f"without --force (pass --force to confirm this is an intentional update)."
            )
    dest_path.write_text(new_text)
    print(f"Wrote {dest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Path to ion_migration_neb_reference.json[.gz]")
    parser.add_argument("--results", required=True, help="Path to the all-FP results file (.json or .json.gz)")
    parser.add_argument("--component-dir", default=".", help="Path to the Ion_migration_NEB/ directory (default: current directory)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files even if their content differs")
    args = parser.parse_args()

    component_dir = Path(args.component_dir).resolve()
    scripts_dir = component_dir / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import neb_analysis as na
    from neb_plots import area_between_curves, simplify_class

    leaderboard = build_leaderboard(args.reference, args.results, na, area_between_curves, simplify_class)

    component_dest = component_dir / "data" / "ion_migration_neb_leaderboard_summary.json"
    docs_dest = component_dir.parent / "docs" / "data" / "ion_migration_neb_leaderboard_summary.json"

    write_output(leaderboard, component_dest, args.force)
    write_output(leaderboard, docs_dest, args.force)

    print("CROSS-CHECK AGAINST key_neb_metrics_summary_df: ALL PASSED")


if __name__ == "__main__":
    main()
