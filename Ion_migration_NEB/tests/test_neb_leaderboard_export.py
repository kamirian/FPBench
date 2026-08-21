"""Regression tests for the NEB leaderboard's JSON-driven rendering
(scripts/export_neb_leaderboard.py -> data/ and docs/data/ ->
docs/ion-migration-neb.html). Confirms the reproducibility fix actually
holds: the page fetches real data rather than hardcoded rows, both shipped
JSON copies are identical, and every field the page's JavaScript reads
really exists in the JSON it reads it from.

Static/structural checks only -- does not execute a browser or JavaScript.
Run with: python tests/test_neb_leaderboard_export.py (from Ion_migration_NEB/)
"""

import json
import re
import sys
from pathlib import Path

COMPONENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = COMPONENT_DIR.parent

COMPONENT_JSON = COMPONENT_DIR / "data" / "ion_migration_neb_leaderboard_summary.json"
DOCS_JSON = REPO_ROOT / "docs" / "data" / "ion_migration_neb_leaderboard_summary.json"
HTML_PAGE = REPO_ROOT / "docs" / "ion-migration-neb.html"

EXPECTED_FP_ORDER = [
    "MACE-MP0_medium", "CHGNET", "M3GNET_pes", "UMA_s1_p1",
    "M3GNET_matpes_PBE", "TensorNET_matpes_PBE", "MACE_matpes_pbe",
]
MODEL_NUMERIC_FIELDS = [
    "n_nonconverged", "n_total",
    "barrier_mae_full_eV", "barrier_rmse_full_eV",
    "barrier_mae_static_eV", "barrier_rmse_static_eV",
    "endpoint_energy_diff_mae_eV", "endpoint_energy_diff_rmse_eV",
    "endpoint_ranking_agreement_pct", "energy_profile_shape_agreement_pct",
]

results = []


def check(name, condition):
    results.append((name, bool(condition)))
    print(("PASS" if condition else "FAIL") + f"  {name}")


# -- 1. both files exist -----------------------------------------------------
check("1. component data/ leaderboard JSON exists", COMPONENT_JSON.exists())
check("1. docs/data/ leaderboard JSON (GitHub Pages copy) exists", DOCS_JSON.exists())

if COMPONENT_JSON.exists() and DOCS_JSON.exists():
    component_bytes = COMPONENT_JSON.read_bytes()
    docs_bytes = DOCS_JSON.read_bytes()

    # -- 2. byte-identical -----------------------------------------------------
    check("2. component and docs/ leaderboard JSON are byte-identical", component_bytes == docs_bytes)

    # -- 3. both valid JSON ------------------------------------------------------
    try:
        component_data = json.loads(component_bytes)
        docs_data = json.loads(docs_bytes)
        check("3. both files are valid JSON", True)
    except json.JSONDecodeError as e:
        check(f"3. both files are valid JSON (error: {e})", False)
        component_data = docs_data = None

    if component_data is not None:
        # -- 4. seven FP records, correct order -----------------------------------
        check("4. fp_order has exactly 7 entries", len(component_data.get("fp_order", [])) == 7)
        check("4. models has exactly 7 records", len(component_data.get("models", [])) == 7)
        check("4. fp_order matches the canonical FP order",
              component_data.get("fp_order") == EXPECTED_FP_ORDER)
        model_keys = [m["fp_key"] for m in component_data.get("models", [])]
        check("4. every fp_order entry has a matching models record, no duplicates",
              sorted(model_keys) == sorted(EXPECTED_FP_ORDER) and len(model_keys) == len(set(model_keys)))

        # -- 5. every model record carries every field the page's JS reads -------
        all_fields_present = all(
            all(field in m for field in MODEL_NUMERIC_FIELDS) and "display_name" in m
            for m in component_data.get("models", [])
        )
        check("5. every model record has all fields the page's JS reads (fp_key, display_name, "
              "and the 10 numeric metric fields)", all_fields_present)

# -- 6. the HTML page fetches the JSON and builds rows from it, rather than --
#      shipping hardcoded scientific rows --------------------------------------
if HTML_PAGE.exists():
    html = HTML_PAGE.read_text()

    check("6. page fetches data/ion_migration_neb_leaderboard_summary.json at runtime",
          "fetch('data/ion_migration_neb_leaderboard_summary.json'" in html)

    # The static markup's <tbody> for table.nebtable must be empty -- rows are
    # only ever inserted by JS, never duplicated as hardcoded <tr> markup.
    m = re.search(r'<table class="dt nebtable">.*?<tbody>\s*</tbody>', html, re.S)
    check("6. table.nebtable's static <tbody> is empty (no hardcoded rows shipped in the HTML)",
          m is not None)

    # No literal FP-specific numeric leaderboard values (data-val="...") should
    # appear anywhere in the static markup outside the <script> block itself --
    # a resurfacing of hardcoded rows would show up here.
    body_only = html.split("<script>")[0]
    check("6. no data-val attributes remain in the static markup (rows are JS-only)",
          "data-val=" not in body_only)

    # -- 7. every JSON field name the JS reads is present in the JS source ------
    script = html.split("<script>")[1].split("</script>")[0] if "<script>" in html else ""
    fields_referenced = all(f"'{field}'" in script or f'"{field}"' in script or f".{field}" in script
                             for field in MODEL_NUMERIC_FIELDS)
    check("7. every model numeric field name is referenced in the page's JavaScript "
          "(rendering can't silently drop a column)", fields_referenced)
    check("7. fp_key and display_name are referenced in the page's JavaScript",
          ("fp_key" in script) and ("display_name" in script))

    # -- 8. fixed direction arrows and legend are still present ------------------
    check("8. fixed lower-is-better / higher-is-better arrow legend is still present",
          ("lower is better" in html) and ("higher is better" in html))
    check("8. every <th data-dir=...> header still carries an explicit direction",
          len(re.findall(r'<th data-dir="(?:low|high)"', html)) == 6)

    # -- 9. error handling: a visible error path exists, not a silent fallback --
    check("9. page defines a visible error-display path for a failed fetch",
          "showLeaderboardError" in script and 'leaderboard-status' in html)
else:
    check("6-9. docs/ion-migration-neb.html exists", False)


# -- summary ------------------------------------------------------------------
n_pass = sum(1 for _, ok in results if ok)
print(f"\n{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("FAILURES:")
    for name, ok in results:
        if not ok:
            print("  -", name)
    sys.exit(1)
sys.exit(0)
