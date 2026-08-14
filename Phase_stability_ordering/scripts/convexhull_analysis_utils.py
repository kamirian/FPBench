"""
convexhull_analysis_utils.py
────────────────────────────
Utility functions for convex hull and ordering analysis of MLIP vs DFT.

Sections
--------
A. Geometry helpers       – tie-line projections, fractions
B. Data builders          – build_clean_*, build_points_for_system, normalize_to_endpoints
C. Convex hull             – lower_hull, plot_normalized_with_hulls (both styles)
D. Energy metrics          – mae_rmse, prepare_points_for_metrics, get_system_metrics
E. Agreement metrics       – compute_gp_same_frac, compute_basebest_same_frac, etc.
F. Aggregate summary       – create_aggregate_summary
G. Structure comparison    – compute_rmsd_table
H. Ordering metrics        – get_Epa, compute_metrics_one_group, summarize_one_potential

Sections A-H above are the original data model (all_mlips_clean_*, endpoints_*,
ordering_merged). Sections I-N below are the standardized FPBench data model
that replaces it in the analysis notebook: explicit candidate identity shared
between DFT and every FP, explicit success/missing status (no silent structure
fallback), and corrected metrics (pooled ground-state agreement excluding
single-phase compositions, RMSD denominators that include calculation/mapping
failures, the manuscript's exact ranking-error formula). Sections A-H remain in
place because the Section-4 (single-system) plotting functions in Section C
still consume the old shape; the standardized-data builders and Section-4
plotting are bridged via an adapter in the notebook rather than rewriting ~900
lines of matplotlib code that was never reported as incorrect.

I. Standardized data model    – phase_id_of, build_dft_hull, build_fp_hull,
                               build_dft_ordering, build_fp_ordering
J. Hull metrics               – ground_state_agreement_pooled,
                               within_phase_hull_min_agreement, global_hull_min_agreement
K. Structure comparison        – compute_structure_rmsd, summarize_structure_rmsd
L. Ordering metrics           – is_misranked_pair, compute_ordering_group_metrics,
                               compute_ordering_summary
M. Data-structure demo        – demonstrate_hull_system, demonstrate_ordering_group
N. Data validation summary    – validate_phase_stability_ordering_results
O. One-call manuscript loader – load_all_convexhull_ordering_data (wraps FORMAT A/B
                               loading + legacy shape + the standardized data model in
                               one call, so the notebook's own data-loading cell stays short)
P. Hull metrics demo (plots) – demonstrate_hull_metrics and its plot_*_demo pieces:
                               same relax/static hull-pair panel as plot_hull_pair,
                               with one phase/composition emphasized and a checkmark/X
                               badge, so a reader can see concretely what a Section 4
                               table cell means -- the agreement checks are identical
                               to ground_state_agreement_pooled / within_phase_hull_min_
                               agreement / global_hull_min_agreement (single-phase /
                               single-composition exclusion included)
Q. Ordering metrics demo     – demonstrate_ordering_ranking / plot_ranking_schematic:
                               DFT-vs-FP energy-level schematic for one ordering group,
                               using the identical is_misranked_pair as the Section 5 table
R. Results-table builders     – build_combined_hull_table, build_combined_ordering_table,
                               build_rmsd_table (one call each, hull/ordering/RMSD)
S. Public builder             – build_phase_stability_ordering_results: normalizes
                               user-supplied DFT reference data + FP results (plain
                               Python structures, not FPBench's historical FORMAT A/
                               FORMAT B file layout) into the same standardized FPBench
                               data structures Sections I-R already consume, for
                               analyzing another dataset with this benchmark's own
                               metrics/table functions unchanged
T. Standardized files          – load_standardized_reference, load_standardized_results,
                               merge_phase_stability_ordering_fp_results: reads/writes
                               the two canonical manuscript-reproduction data products
                               (data/phase_stability_ordering_reference.json.gz and
                               data/phase_stability_ordering_results_standardized.json.gz)
                               and combines standardized single-FP result fragments into
                               the merged "models" mapping build_phase_stability_ordering_
                               results accepts as fp_results
"""

from __future__ import annotations
import re, copy, json, gzip, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict

from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor

# Matplotlib markers rendered as strokes only (no fill path).
# facecolors="none" in scatter makes these invisible, so hollow treatment is skipped for them.
_LINE_ONLY_MARKERS = frozenset({"x", "+", "|", "_", "1", "2", "3", "4"})

# Default phase color palette: tab10 without orange (C1) and gray (C7).
# Avoids clashing with typical hull line colors (darkorange, black).
# Override via phase_colors= in plot_normalized_with_hulls / plot_hull_pair.
_PHASE_COLORS_DEFAULT = ["C0", "C2", "C3", "C4", "C5", "C6", "C8", "C9"]
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher

# ═══════════════════════════════════════════════════════════════════════════════
# A. Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def frac_dict(C: Composition) -> dict:
    """Atomic fractions: Bi2Te3 → {'Bi': 0.4, 'Te': 0.6}"""
    d = C.get_el_amt_dict()
    total = float(sum(d.values()))
    return {el: amt / total for el, amt in d.items()}


def extract_base(key: str) -> str:
    """'mp-1198150_Bi3+12_Sb3+4_Se2-24' → 'mp-1198150'"""
    return key.split("_", 1)[0]


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_system_name(name: str) -> str:
    """'GeSe2_SiSe2' -> 'GeSe₂–SiSe₂' (subscript digits, en-dash separator)."""
    result = re.sub(r"\d+", lambda m: m.group().translate(_SUBSCRIPT_DIGITS), name)
    return result.replace("_", "–")


def parse_endmembers(system_name: str):
    """'Bi2Se3_Sb2Se3' → (frac_dict(Bi2Se3), frac_dict(Sb2Se3))"""
    a, b = system_name.split("_", 1)
    return frac_dict(Composition(a)), frac_dict(Composition(b))


def structure_frac_dict(struct: Structure):
    """Returns per-atom element fraction dict and number of sites."""
    d = struct.composition.get_el_amt_dict()
    total = float(sum(d.values()))
    return {el: amt / total for el, amt in d.items()}, struct.num_sites


def to_vec(fd: dict, elements) -> np.ndarray:
    return np.array([fd.get(el, 0.0) for el in elements], dtype=float)


def compute_x_projection(frC: dict, frA: dict, frB: dict) -> float | None:
    """Project composition C onto the A–B tie-line. Returns x in [0,1]."""
    elements = sorted(set(frA) | set(frB) | set(frC))
    A = to_vec(frA, elements)
    B = to_vec(frB, elements)
    C = to_vec(frC, elements)
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-15:
        return None
    return float(np.dot(C - A, AB) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# B. Data builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_clean_from_flat(flat_relax: dict, flat_static: dict) -> tuple[dict, dict]:
    """
    Build clean relax and static dicts from the flat format produced by the
    new run notebooks / generator.

    Input format (per MLIP, flat dict keyed by sid):
        {
          "tie_line||system||composition||ordered_name": {
              "system":               "SnSe_Bi2Se3",
              "composition":          "mp-12345_...",
              "ordered_name":         "ordered_0",
              "DFT_final_energy":     -123.45,
              "DFT_initial_structure": {...},   # relax only
              "DFT_final_structure":  {...},
              "MLIP_result": {
                  "mode": "relaxation",         # or "static"
                  "energy": -122.0,
                  "relaxed_structure": {...},   # relax only
              }
          }
        }

    Returns
    -------
    (all_clean_relax, all_clean_static)
      each: {system: {key: {structure, n_atoms, composition, sid,
                             MLIP_energy, MLIP_energy/atom, DFT_energy, DFT_energy/atom}}}
    """
    def _parse(entries: dict, mode: str) -> dict:
        out: dict[str, dict] = {}
        for sid, entry in entries.items():
            # support both top-level keys (old format) and source_keys (new flat format)
            sk = entry.get("source_keys", {})
            system = entry.get("system") or sk.get("system")
            if not system:
                # last resort: parse from sid "tie_line||system||comp||ordered"
                parts = sid.split("||")
                system = parts[1] if len(parts) >= 2 else None
            if not system:
                continue
            mlip_res = entry.get("MLIP_result", {})
            # For relax: use the MLIP-relaxed structure; for static: use DFT final
            if mode == "relax":
                struct_dict = mlip_res.get("relaxed_structure") or entry.get("DFT_initial_structure")
            else:
                struct_dict = entry.get("DFT_final_structure")
            if struct_dict is None:
                continue
            try:
                struct = Structure.from_dict(struct_dict)
            except Exception:
                continue
            n      = len(struct.sites)
            mlip_E = mlip_res.get("energy")
            dft_E  = entry.get("DFT_final_energy")
            if mlip_E is None or dft_E is None:
                continue
            comp_name    = entry.get("composition") or sk.get("composition", "")
            ordered_name = entry.get("ordered_name") or sk.get("ordered_name", "")
            key          = f"{comp_name}_{ordered_name}"
            out.setdefault(system, {})[key] = {
                "structure":        struct,
                "n_atoms":          n,
                "composition":      struct.composition,
                "sid":              comp_name.split("_")[0] if comp_name else (sid.split("||")[0] if "||" in sid else sid),
                "MLIP_energy":      float(mlip_E),
                "MLIP_energy/atom": float(mlip_E) / n,
                "DFT_energy":       float(dft_E),
                "DFT_energy/atom":  float(dft_E) / n,
            }
        return out

    return _parse(flat_relax, "relax"), _parse(flat_static, "static")


def load_vasp_endpoint_dft_energies(old_endpoints_json_path: str) -> dict:
    """
    Extract VASP-run DFT endpoint energies from the old nested endpoint JSON
    (ALL_MLIP_endpoints_merged.json).

    Returns {mp_id_number: DFT_final_energy_total}  e.g. {"541837": -20.1989}

    These energies come from the same VASP settings as the hull interior entries,
    so they can be used as the DFT reference when normalising the convex hull.
    The MP-database DFT_energy stored in the new flat endpoint JSON uses different
    settings and should NOT be mixed with tie-line VASP energies.
    """
    import json as _json
    with open(old_endpoints_json_path) as f:
        ep_old = _json.load(f)
    relax = ep_old.get("MLIP_full_relaxation_initials_endpoints.json", {})
    # Take DFT energies from the first MLIP (they are MLIP-independent)
    first_mlip = next(iter(relax.values()), {})
    out = {}
    for id_key, d in first_mlip.items():
        mp_num = id_key.split("_")[-1]   # "Bi_Se_entries_541837" → "541837"
        e = d.get("DFT_final_energy")
        if e is not None:
            out[mp_num] = float(e)
    return out


def build_endpoints_from_flat(flat_ep: dict,
                               vasp_dft_energies: dict | None = None) -> dict:
    """
    Build endpoints dict from the flat format produced by the new run notebooks.

    Input format (flat dict keyed by entry_id):
        {
          "mp-12345-GGA": {
              "entry_id":    "mp-12345-GGA",
              "composition": "Bi2Se3",
              "DFT_energy":  -100.0,          ← MP-database energy (may differ from VASP run)
              "DFT_structure": {...},
              "MLIP_result":   {"energy": ..., "relaxed_structure": {...}}
          }
        }

    Parameters
    ----------
    vasp_dft_energies : optional dict {mp_id_number: DFT_final_energy}
        VASP-run DFT energies from the old endpoint JSON (produced by
        load_vasp_endpoint_dft_energies).  When provided, these override the
        MP-database DFT_energy for each endpoint so that the DFT reference
        is consistent with the tie-line interior entries.

    Returns {sid: {structure, n_atoms, composition, sid,
                   MLIP_energy, MLIP_energy/atom, DFT_energy, DFT_energy/atom,
                   endpoint_id}}
    """
    out: dict[str, dict] = {}
    for eid, entry in flat_ep.items():
        mlip_res    = entry.get("MLIP_result", {})
        struct_dict = mlip_res.get("relaxed_structure") or entry.get("DFT_structure")
        if struct_dict is None:
            continue
        try:
            struct = Structure.from_dict(struct_dict)
        except Exception:
            continue
        n      = len(struct.sites)
        mlip_E = mlip_res.get("energy")
        dft_E  = entry.get("DFT_energy")
        if mlip_E is None or dft_E is None:
            continue
        sid = entry.get("entry_id", eid)
        if not sid.startswith("mp-"):
            sid = f"mp-{sid}"
        # Normalize: strip GGA/LDA/SCAN/r2SCAN suffixes (e.g. "mp-699208-GGA" → "mp-699208")
        sid = re.sub(r"-(GGA|LDA|SCAN|r2SCAN|PBE|PBEsol)(\+U)?$", "", sid)
        # Override DFT energy with VASP-run value when available
        if vasp_dft_energies is not None:
            mp_num = sid.split("-")[-1]   # "mp-541837" → "541837"
            if mp_num in vasp_dft_energies:
                dft_E = vasp_dft_energies[mp_num]
        out[sid] = {
            "endpoint_id":      eid,
            "structure":        struct,
            "n_atoms":          n,
            "composition":      struct.composition,
            "sid":              sid,
            "MLIP_energy":      float(mlip_E),
            "MLIP_energy/atom": float(mlip_E) / n,
            "DFT_energy":       float(dft_E),
            "DFT_energy/atom":  float(dft_E) / n,
        }
    return out


def build_clean_relax(all_merged: dict) -> dict:
    """
    Extract MLIP full-relaxation results from the raw merged JSON.

    Returns
    -------
    dict: all_mlips_clean[mlip][system][key] = {structure, n_atoms,
          composition, sid, MLIP_energy, MLIP_energy/atom,
          DFT_energy, DFT_energy/atom}
    """
    out = {}
    for mlip, data in all_merged.items():
        out[mlip] = {}
        for vname, vdata in data["MLIP_full_relaxation_initials.json"].items():
            for system_name, sdata in vdata.items():
                out[mlip].setdefault(system_name, {})
                for comp_name, cdata in sdata.items():
                    for struct_name, sinfo in cdata.items():
                        key = f"{comp_name}_{struct_name}"
                        struct = Structure.from_dict(
                            sinfo["MLIP_result"]["relaxed_structure"]
                        )
                        n = len(struct.sites)
                        mlip_E = sinfo["MLIP_result"]["energy"]
                        dft_E  = sinfo["DFT_final_structure_data"]["dft_Ef"]
                        out[mlip][system_name][key] = {
                            "structure":        struct,
                            "n_atoms":          n,
                            "composition":      struct.composition,
                            "sid":              comp_name.split("_")[0],
                            "MLIP_energy":      mlip_E,
                            "MLIP_energy/atom": mlip_E / n,
                            "DFT_energy":       dft_E,
                            "DFT_energy/atom":  dft_E / n,
                        }
    return out


def build_clean_static(all_merged: dict, all_mlips_clean_relax: dict) -> dict:
    """
    Extract MLIP static results. Uses relaxed structure from the relax run
    for geometry (atom count) consistency.
    """
    out = {}
    for mlip, data in all_merged.items():
        out[mlip] = {}
        for vname, vdata in data["MLIP_static_finals.json"].items():
            for system_name, sdata in vdata.items():
                out[mlip].setdefault(system_name, {})
                for comp_name, cdata in sdata.items():
                    for struct_name, sinfo in cdata.items():
                        key = f"{comp_name}_{struct_name}"
                        # Use relaxed structure for geometry
                        try:
                            struct = Structure.from_dict(
                                all_merged[mlip]["MLIP_full_relaxation_initials.json"]
                                [vname][system_name][comp_name][struct_name]
                                ["MLIP_result"]["relaxed_structure"]
                            )
                        except (KeyError, TypeError):
                            continue
                        n = len(struct.sites)
                        mlip_E = sinfo["MLIP_result"]["energy"]
                        dft_E  = sinfo["DFT_final_structure_data"]["dft_Ef"]
                        out[mlip][system_name][key] = {
                            "structure":        struct,
                            "n_atoms":          n,
                            "composition":      struct.composition,
                            "sid":              comp_name.split("_")[0],
                            "MLIP_energy":      mlip_E,
                            "MLIP_energy/atom": mlip_E / n,
                            "DFT_energy":       dft_E,
                            "DFT_energy/atom":  dft_E / n,
                        }
    return out


def build_endpoints_relax(all_endpoints_merged: dict) -> dict:
    """endpoints_by_mlip[mlip][sid] = {structure, n_atoms, ...}"""
    out = {}
    for mlip, data in all_endpoints_merged["MLIP_full_relaxation_initials_endpoints.json"].items():
        out.setdefault(mlip, {})
        for id_key, id_data in data.items():
            sid = "mp-" + id_key.split("_")[-1]
            struct = Structure.from_dict(id_data["MLIP_result"]["relaxed_structure"])
            n = len(struct.sites)
            mlip_E = id_data["MLIP_result"]["energy"]
            dft_E  = id_data["DFT_final_energy"]
            out[mlip][sid] = {
                "endpoint_id":      id_key,
                "structure":        struct,
                "n_atoms":          n,
                "composition":      struct.composition,
                "sid":              sid,
                "MLIP_energy":      mlip_E,
                "MLIP_energy/atom": mlip_E / n,
                "DFT_energy":       dft_E,
                "DFT_energy/atom":  dft_E / n,
            }
    return out


def build_endpoints_static(all_endpoints_merged: dict) -> dict:
    """endpoints_by_mlip_static[mlip][sid] = {structure, n_atoms, ...}"""
    out = {}
    for mlip, data in all_endpoints_merged["MLIP_finals_static_endpoints.json"].items():
        out.setdefault(mlip, {})
        for id_key, id_data in data.items():
            sid = "mp-" + id_key.split("_")[-1]
            struct = Structure.from_dict(id_data["mp_structure"])
            n = len(struct.sites)
            mlip_E = id_data["MLIP_result"]["energy"]
            dft_E  = id_data["DFT_final_energy"]
            out[mlip][sid] = {
                "endpoint_id":      id_key,
                "structure":        struct,
                "n_atoms":          n,
                "composition":      struct.composition,
                "sid":              sid,
                "MLIP_energy":      mlip_E,
                "MLIP_energy/atom": mlip_E / n,
                "DFT_energy":       dft_E,
                "DFT_energy/atom":  dft_E / n,
            }
    return out


def merge_endpoints_into_hull(all_mlips_clean: dict, endpoints_by_mlip: dict) -> dict:
    """
    Add endpoint entries into the hull dict so they appear on the tie-line plots.
    Endpoints are added to every system that contains their mp-id.
    """
    out = copy.deepcopy(all_mlips_clean)

    # Build sid → systems map
    sid_to_systems: dict[str, dict[str, set]] = {}
    for mlip, systems in all_mlips_clean.items():
        sid_to_systems[mlip] = {}
        for system_name, entries in systems.items():
            for entry in entries.values():
                sid = entry.get("sid")
                if sid:
                    sid_to_systems[mlip].setdefault(sid, set()).add(system_name)

    for mlip, sid_map in endpoints_by_mlip.items():
        if mlip not in out:
            continue
        for sid, sid_data in sid_map.items():
            struct = sid_data.get("structure")
            if struct is None:
                continue
            ep_id   = sid_data.get("endpoint_id", "endpoint")
            new_key = f"{sid}__{ep_id}"
            for system_name in sid_to_systems.get(mlip, {}).get(sid, []):
                # Only add if the endpoint composition sits at an end-member
                # position (x≈0 or x≈1). This prevents interior-composition
                # prototypes from being mis-labelled as endpoints.
                try:
                    frA, frB = parse_endmembers(system_name)
                    frC, _   = structure_frac_dict(struct)
                    x = compute_x_projection(frC, frA, frB)
                except Exception:
                    x = None
                if x is None or not (x <= 0.05 or x >= 0.95):
                    continue
                out[mlip][system_name].setdefault(new_key, sid_data)

    return out


def build_dft_structures(all_DFT_merged: dict) -> dict:
    """all_DFT_clean[system][key] = Structure (DFT final structure)"""
    out = {}
    for vname, vdata in all_DFT_merged.items():
        for system_name, sdata in vdata.items():
            out.setdefault(system_name, {})
            for comp_name, cdata in sdata.items():
                for struct_name, sinfo in cdata.items():
                    key = f"{comp_name}_{struct_name}"
                    out[system_name][key] = Structure.from_dict(sinfo["final_structure"])
    return out


def merge_dft_endpoints(all_DFT_clean: dict, endpoints_by_DFT: dict) -> dict:
    """Add DFT endpoint structures into all_DFT_clean for RMSD comparison.

    endpoints_by_DFT must be {id_key: structure} where id_key is the full
    endpoint identifier (e.g. 'Bi_Se_entries_541837'), matching the format
    used on the MLIP side by merge_endpoints_into_hull.
    """
    out = copy.deepcopy(all_DFT_clean)
    sid_to_systems: dict[str, set] = {}
    for system_name, entries in all_DFT_clean.items():
        for key in entries:
            sid = key.split("_")[0]
            sid_to_systems.setdefault(sid, set()).add(system_name)

    for id_key, struct in endpoints_by_DFT.items():
        # New flat format: id_key = "mp-699208-GGA"  → sid = "mp-699208-GGA"
        # Old format:      id_key = "Bi_Se_entries_541837" → sid = "mp-541837"
        if id_key.startswith("mp-"):
            sid = re.sub(r"-(GGA|LDA|SCAN|r2SCAN|PBE|PBEsol)(\+U)?$", "", id_key)
        else:
            sid = "mp-" + id_key.split("_")[-1]
        new_key    = f"{sid}__{id_key}"
        struct_obj = Structure.from_dict(struct) if isinstance(struct, dict) else struct
        for system_name in sid_to_systems.get(sid, []):
            out[system_name].setdefault(new_key, struct_obj)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# C. Convex hull plotting
# ═══════════════════════════════════════════════════════════════════════════════

def build_points_for_system(all_merged: dict, mlip: str, system_name: str,
                             round_x: int = 8, clip_x: bool = True) -> list[dict]:
    """
    Returns list of point dicts:
      {key, x, is_endpoint, E_mlip, E_dft}
    """
    dsys = all_merged[mlip][system_name]
    frA, frB = parse_endmembers(system_name)
    points = []
    for key, entry in dsys.items():
        struct = entry.get("structure")
        if struct is None:
            continue
        frC, _ = structure_frac_dict(struct)
        x = compute_x_projection(frC, frA, frB)
        if x is None:
            continue
        if clip_x:
            x = min(1.0, max(0.0, float(x)))
        if round_x is not None:
            x = round(float(x), round_x)
        points.append({
            "key":         key,
            "x":           x,
            "is_endpoint": ("__" in key),
            "E_mlip":      entry.get("MLIP_energy/atom"),
            "E_dft":       entry.get("DFT_energy/atom"),
        })
    return points


def pick_endpoints_by_x(points: list[dict]):
    """Return the lowest-x and highest-x points (by lowest energy at ties)."""
    min_x = min(p["x"] for p in points)
    max_x = max(p["x"] for p in points)
    at_min = [p for p in points if abs(p["x"] - min_x) < 1e-5]
    at_max = [p for p in points if abs(p["x"] - max_x) < 1e-5]
    p0 = min(at_min, key=lambda p: p["E_mlip"] if p["E_mlip"] is not None else float("inf"))
    p1 = min(at_max, key=lambda p: p["E_mlip"] if p["E_mlip"] is not None else float("inf"))
    return p0, p1


def normalize_to_endpoints(points: list[dict]):
    """
    Add Erel_mlip and Erel_dft (formation-energy-like, relative to tie-line).
    Returns (points, p0, p1).
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")
    p0, p1 = pick_endpoints_by_x(points)
    x0, x1 = p0["x"], p1["x"]
    E0m, E1m = p0["E_mlip"], p1["E_mlip"]
    E0d, E1d = p0["E_dft"],  p1["E_dft"]
    dx = (x1 - x0) if abs(x1 - x0) > 1e-6 else 1.0
    for p in points:
        x = p["x"]
        if all(v is not None for v in [E0m, E1m, p["E_mlip"]]):
            p["Eref_mlip"] = E0m + (x - x0) * (E1m - E0m) / dx
            p["Erel_mlip"] = p["E_mlip"] - p["Eref_mlip"]
        if all(v is not None for v in [E0d, E1d, p["E_dft"]]):
            p["Eref_dft"] = E0d + (x - x0) * (E1d - E0d) / dx
            p["Erel_dft"] = p["E_dft"] - p["Eref_dft"]
    return points, p0, p1


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lower_hull(points_xy: list[tuple]) -> list[tuple]:
    """Lower convex hull (stable compositions) from list of (x, y) pairs."""
    collapsed = {}
    for x, y in points_xy:
        if x not in collapsed or y < collapsed[x]:
            collapsed[x] = y
    pts = sorted(collapsed.items())
    H = []
    for p in pts:
        while len(H) >= 2 and _cross(H[-2], H[-1], p) <= 0:
            H.pop()
        H.append(p)
    return H


def _collect_legend(axes):
    """Deduplicate legend handles across multiple axes.

    Deduplication uses (label, edgecolor) so that two phases that happen to
    share the same label text (e.g. same reduced formula) are still kept as
    separate entries, while the same handle appearing in both subplots is
    collapsed to one entry.  Tuple handles (HandlerTuple pairs) are keyed by
    the edgecolor of their first element.
    """
    seen = OrderedDict()  # (label, edgecolor_str) → (handle, label)
    for ax in axes:
        if hasattr(ax, "_full_legend_handles"):
            pairs = zip(ax._full_legend_handles, ax._full_legend_labels)
        else:
            pairs = zip(*ax.get_legend_handles_labels())
        for h, l in pairs:
            try:
                h0 = h[0] if isinstance(h, tuple) else h
                ec = str(h0.get_markeredgecolor())
            except Exception:
                ec = ""
            key = (l, ec)
            if key not in seen:
                seen[key] = (h, l)
    return [v[0] for v in seen.values()], [v[1] for v in seen.values()]


def _collect_legend_grouped(axes, phase_ncol=3, marker_ncol=None, hull_ncol=None):
    """Collect and arrange legend handles into three visually separate rows.

    Layout produced (with default phase_ncol=3):
        Row(s): [MLIP marker] [DFT marker]  [blank …]   ← marker_ncol wide
        Row(s): [MLIP hull]   [DFT hull]    [blank …]   ← hull_ncol wide
        Row(s): [phase 1]     [phase 2]     [phase 3]   ← phase_ncol wide
                [phase 4]     [phase 5]     [phase 6]
        …
    Pass fig.legend(…, ncol=phase_ncol) so the grid aligns.

    marker_ncol / hull_ncol default to phase_ncol when None.
    """
    if marker_ncol is None:
        marker_ncol = phase_ncol
    if hull_ncol is None:
        hull_ncol = phase_ncol

    def _ec(h):
        try:
            h0 = h[0] if isinstance(h, tuple) else h
            return str(h0.get_markeredgecolor())
        except Exception:
            return ""

    def _dedup(attr_h, attr_l):
        seen: OrderedDict = OrderedDict()
        for ax in axes:
            for h, l in zip(getattr(ax, attr_h, []), getattr(ax, attr_l, [])):
                seen.setdefault((l, _ec(h)), (h, l))
        return [v[0] for v in seen.values()], [v[1] for v in seen.values()]

    blank = Line2D([], [], linestyle="", marker="")

    def _pad(handles, labels, ncol):
        p = (-len(handles)) % ncol
        return handles + [blank] * p, labels + [""] * p

    marker_h, marker_l = _dedup("_legend_marker_handles", "_legend_marker_labels")
    hull_h,   hull_l   = _dedup("_legend_hull_handles",   "_legend_hull_labels")
    phase_h,  phase_l  = _dedup("_legend_phase_handles",  "_legend_phase_labels")

    # Fall back to combined method group if separate groups were not stored
    if not marker_h and not hull_h:
        marker_h, marker_l = _dedup("_legend_method_handles", "_legend_method_labels")
        marker_ncol = phase_ncol

    marker_h, marker_l = _pad(marker_h, marker_l, marker_ncol)
    hull_h,   hull_l   = _pad(hull_h,   hull_l,   hull_ncol)

    return marker_h + hull_h + phase_h, marker_l + hull_l + phase_l


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_composition_label(name: str) -> str:
    """'GeSe2' -> 'GeSe₂' (digits rendered as unicode subscripts)."""
    return re.sub(r"\d+", lambda m: m.group().translate(_SUBSCRIPT_DIGITS), name)


def format_spacegroup_symbol(symbol: str) -> str:
    """
    Render a pymatgen space-group symbol with proper crystallographic
    typesetting for matplotlib mathtext:
      - '-N' (bar over digit N, e.g. inversion/rotoinversion axes) becomes
        an overline:      'R-3m'   -> 'R$\\mathrm{\\overline{3}}$m'
                           'I-42d'  -> 'I$\\mathrm{\\overline{4}}$2d'
      - '_N' (screw-axis subscript) becomes a true subscript:
                           'P2_1/c' -> 'P2$_{1}$/c'
    Both can appear in the same symbol, e.g. 'P-42_1m' -> 'P$\\mathrm{\\overline{4}}$2$_{1}$m'.
    Everything else is left as plain text.
    """
    if not symbol:
        return symbol

    def _sub(m):
        bar_digit, sub_digit = m.group(1), m.group(2)
        if bar_digit is not None:
            return r"$\mathrm{\overline{%s}}$" % bar_digit
        return r"$_{%s}$" % sub_digit

    return re.sub(r"-(\d)|_(\d)", _sub, symbol)


def plot_normalized_with_hulls(
    all_merged, mlip, system_name,
    title=None, dodge=0.01,
    figsize=(5.2, 3.8), s=35, alpha=0.9, textsize=11,
    show_legend=True, static_tag=None, ax=None,
    mlip_marker="o", dft_marker="s",
    mlip_hollow=False, dft_hollow=False,
    mlip_alpha=None,   # alpha for MLIP scatter points; None → falls back to alpha
    dft_alpha=None,    # alpha for DFT scatter points;  None → falls back to alpha
    mlip_display=None,
    label_by="mpid",  # "mpid" | "composition" | "spacegroup" | "composition+spacegroup"
    hull_lw=1.0,
    show_zero_line=True,
    marker_label=None,       # None → "MLIP / DFT" combined; "separate" → two entries; any str → use as label
    mlip_hull_color=None,    # explicit color for MLIP hull line (None = matplotlib default)
    dft_hull_color=None,     # explicit color for DFT hull line  (None = matplotlib default)
    hull_alpha=1.0,          # opacity of convex hull lines (0=invisible, 1=solid)
    zero_line_alpha=0.35,    # opacity of the dashed E_f=0 reference line
    phase_colors=None,       # list of colors for phases; None → _PHASE_COLORS_DEFAULT
    axis_label_fontweight="normal",  # fontweight for x/y axis labels ("normal", "bold", or numeric)
    mlip_lw=1.0,             # stroke width for MLIP markers (controls + / x thickness)
    dft_lw=1.0,              # stroke width for DFT  markers
    show_endpoint_labels=False,      # show composition names at x=0 / x=1 (tie-line endmembers)
    endpoint_labels=None,            # (left_label, right_label) override; None → derived from system_name
    endpoint_label_fontsize=None,    # None → falls back to textsize
    endpoint_label_fontweight="normal",
    endpoint_label_y=-0.10,          # base vertical position (axes fraction; 0=axis line, negative=below)
    endpoint_label_left_dx=0.0,      # horizontal nudge for the x=0 label (data/atomic-fraction units)
    endpoint_label_left_dy=0.0,      # vertical nudge for the x=0 label (axes-fraction units)
    endpoint_label_right_dx=0.0,     # horizontal nudge for the x=1 label (data/atomic-fraction units)
    endpoint_label_right_dy=0.0,     # vertical nudge for the x=1 label (axes-fraction units)
):
    """
    Scatter + lower-hull plot coloured by mp-id (phases).
    mlip_marker / dft_marker : any matplotlib marker string ("o", "s", "^", "D", "x", "P", …).
    mlip_hollow / dft_hollow : set True for an unfilled (outline-only) marker.
    """
    points = build_points_for_system(all_merged, mlip, system_name)
    points, p0, p1 = normalize_to_endpoints(points)

    mpids = sorted({extract_base(p["key"]) for p in points})

    # Build label map first so color assignment can group by label.
    # For composition/spacegroup labels we pick the ordered structure of each
    # mp-id that sits closest to a pure endmember (min distance to x=0 or x=1),
    # because that structure best represents the parent prototype's composition.
    dsys = all_merged[mlip][system_name]
    _mpid_repr_key: dict[str, str] = {}  # mpid → key of best representative
    if label_by != "mpid":
        _mpid_best_dist: dict[str, float] = {}
        for _p in points:
            _mpid = extract_base(_p["key"])
            _dist = min(_p["x"], 1.0 - _p["x"])
            if _mpid not in _mpid_best_dist or _dist < _mpid_best_dist[_mpid]:
                _mpid_best_dist[_mpid] = _dist
                _mpid_repr_key[_mpid]  = _p["key"]

    mpid_to_label: dict[str, str] = {}
    for _mpid in mpids:
        if label_by == "mpid":
            mpid_to_label[_mpid] = _mpid
            continue
        repr_key = _mpid_repr_key.get(_mpid)
        _entry   = dsys.get(repr_key, {}) if repr_key else {}
        if label_by == "composition":
            _comp = _entry.get("composition")
            mpid_to_label[_mpid] = str(_comp.reduced_formula) if _comp else _mpid
        elif label_by == "spacegroup":
            _struct = _entry.get("structure")
            try:
                sg = _struct.get_space_group_info()[0] if _struct else None
                mpid_to_label[_mpid] = format_spacegroup_symbol(sg) if sg else _mpid
            except Exception:
                mpid_to_label[_mpid] = _mpid
        elif label_by == "composition+spacegroup":
            _comp   = _entry.get("composition")
            _struct = _entry.get("structure")
            comp_str = str(_comp.reduced_formula) if _comp else _mpid
            try:
                sg = _struct.get_space_group_info()[0] if _struct else ""
                sg_fmt = format_spacegroup_symbol(sg)
                mpid_to_label[_mpid] = f"{comp_str} ({sg_fmt})"
            except Exception:
                mpid_to_label[_mpid] = comp_str
        else:
            mpid_to_label[_mpid] = _mpid
        mpid_to_label.setdefault(_mpid, _mpid)

    # If two different mp-ids still share the same display label, append the
    # numeric part of the mp-id to make each label unique.
    if label_by != "mpid":
        from collections import Counter as _Counter
        _lbl_count = _Counter(mpid_to_label.values())
        for _mpid in mpids:
            if _lbl_count[mpid_to_label[_mpid]] > 1:
                _lbl = mpid_to_label[_mpid]
                if _lbl.endswith(")"):
                    mpid_to_label[_mpid] = f"{_lbl[:-1]}, {_mpid})"
                else:
                    mpid_to_label[_mpid] = f"{_lbl} [{_mpid}]"

    # mp-ids that share the same display label get the same color
    _palette = phase_colors if phase_colors is not None else _PHASE_COLORS_DEFAULT
    _label_cidx: dict[str, int] = {}
    _cidx = 0
    color_map: dict[str, str] = {}
    for _mpid in mpids:
        _lbl = mpid_to_label[_mpid]
        if _lbl not in _label_cidx:
            _label_cidx[_lbl] = _cidx
            _cidx += 1
        color_map[_mpid] = _palette[_label_cidx[_lbl] % len(_palette)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _mlip_alpha = mlip_alpha if mlip_alpha is not None else alpha
    _dft_alpha  = dft_alpha  if dft_alpha  is not None else alpha
    for p in points:
        c = color_map[extract_base(p["key"])]
        if p.get("Erel_mlip") is not None:
            # Hollow only for markers that have a fill area; line-only (+ x) are always filled paths
            if mlip_hollow and mlip_marker not in _LINE_ONLY_MARKERS:
                ax.scatter(p["x"] - dodge, p["Erel_mlip"], facecolors="none", edgecolors=c,
                           s=s, alpha=_mlip_alpha, marker=mlip_marker, linewidths=mlip_lw)
            else:
                ax.scatter(p["x"] - dodge, p["Erel_mlip"], c=c, s=s, alpha=_mlip_alpha,
                           marker=mlip_marker, linewidths=mlip_lw)
        if p.get("Erel_dft") is not None:
            if dft_hollow and dft_marker not in _LINE_ONLY_MARKERS:
                ax.scatter(p["x"] + dodge, p["Erel_dft"], facecolors="none", edgecolors=c,
                           s=s, alpha=_dft_alpha, marker=dft_marker, linewidths=dft_lw)
            else:
                # edgecolors is only meaningful for markers with a fill area;
                # matplotlib silently ignores it (and warns) for line-only
                # markers like '+'/'x', so omit it there -- zero visual change.
                _dft_edge_kw = {} if dft_marker in _LINE_ONLY_MARKERS else {"edgecolors": "black"}
                ax.scatter(p["x"] + dodge, p["Erel_dft"], c=c, s=s, alpha=_dft_alpha,
                           marker=dft_marker, linewidths=dft_lw, **_dft_edge_kw)

    _fp_label = mlip_display if mlip_display is not None else mlip

    mlip_xy = [(p["x"], p["Erel_mlip"]) for p in points if p.get("Erel_mlip") is not None]
    dft_xy  = [(p["x"], p["Erel_dft"])  for p in points if p.get("Erel_dft")  is not None]
    _kw_mh = {"color": mlip_hull_color} if mlip_hull_color is not None else {}
    _kw_dh = {"color": dft_hull_color}  if dft_hull_color  is not None else {}
    if len(mlip_xy) >= 2:
        Hm = lower_hull(mlip_xy)
        ax.plot([t[0] for t in Hm], [t[1] for t in Hm], lw=hull_lw, alpha=hull_alpha, label=f"{_fp_label} convex hull", **_kw_mh)
    if len(dft_xy) >= 2:
        Hd = lower_hull(dft_xy)
        ax.plot([t[0] for t in Hd], [t[1] for t in Hd], lw=hull_lw, alpha=hull_alpha, label="DFT convex hull", **_kw_dh)

    # Each color entry shows both markers (MLIP left, DFT right) via HandlerTuple.
    _handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.5)}
    color_handles = []
    color_labels  = []
    for m in mpids:
        c = color_map[m]
        h_m = Line2D([0], [0], marker=mlip_marker, linestyle="",
                     markerfacecolor="none" if (mlip_hollow or mlip_marker in _LINE_ONLY_MARKERS) else c,
                     markeredgecolor=c, markersize=6, markeredgewidth=mlip_lw)
        h_d = Line2D([0], [0], marker=dft_marker, linestyle="",
                     markerfacecolor="none" if (dft_hollow or dft_marker in _LINE_ONLY_MARKERS) else c,
                     markeredgecolor=c, markersize=6, markeredgewidth=dft_lw)
        color_handles.append((h_m, h_d))
        color_labels.append(mpid_to_label.get(m, m))

    # marker_label controls the type-indicator row in the legend:
    #   None or "auto"  → combined entry  "[▽][×] MLIP / DFT"
    #   "separate"      → two separate entries  "[▽] MLIP"  "[×] DFT"
    #   any other str   → combined entry with that exact string as the label
    _h_m = Line2D([0], [0], marker=mlip_marker, linestyle="",
                  markerfacecolor="none" if (mlip_hollow or mlip_marker in _LINE_ONLY_MARKERS) else "gray",
                  markeredgecolor="gray", markersize=6, markeredgewidth=mlip_lw)
    _h_d = Line2D([0], [0], marker=dft_marker, linestyle="",
                  markerfacecolor="none" if (dft_hollow or dft_marker in _LINE_ONLY_MARKERS) else "gray",
                  markeredgecolor="dimgray", markersize=6, markeredgewidth=dft_lw)
    if marker_label == "separate":
        marker_handles = [_h_m, _h_d]
        marker_labels  = [_fp_label, "DFT"]
    else:
        _lbl = marker_label if (marker_label and marker_label not in ("auto", None)) else f"{_fp_label} / DFT"
        marker_handles = [(_h_m, _h_d)]
        marker_labels  = [_lbl]
    hull_handles, hull_labels = ax.get_legend_handles_labels()
    method_handles = marker_handles + hull_handles
    method_labels  = marker_labels + hull_labels
    all_handles = method_handles + color_handles
    all_labels  = method_labels + color_labels
    ax._full_legend_handles    = all_handles
    ax._full_legend_labels     = all_labels
    ax._legend_marker_handles  = marker_handles
    ax._legend_marker_labels   = marker_labels
    ax._legend_hull_handles    = hull_handles
    ax._legend_hull_labels     = hull_labels
    ax._legend_method_handles  = method_handles
    ax._legend_method_labels   = method_labels
    ax._legend_phase_handles   = color_handles
    ax._legend_phase_labels    = color_labels
    ax._legend_handler_map     = _handler_map

    if show_legend:
        ax.legend(all_handles, all_labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=3, fontsize=textsize,
                  handler_map=_handler_map)

    if show_zero_line:
        ax.axhline(0.0, lw=1.0, ls="--", color="black", alpha=zero_line_alpha)
    ax.set_xlabel("Atomic fraction x", fontsize=textsize, fontweight=axis_label_fontweight)
    ax.set_ylabel("$E_f$ (eV/atom)", fontsize=textsize, fontweight=axis_label_fontweight)
    if title is None:
        tag = " (static)" if static_tag else " (full)"
        display = mlip_display if mlip_display is not None else mlip
        title = f"DFT vs {display}{tag}"
    ax.set_title(title, fontsize=textsize + 2)

    if show_endpoint_labels:
        if endpoint_labels is not None:
            _left_lbl, _right_lbl = endpoint_labels
        else:
            _left_name, _right_name = (
                system_name.split("_", 1) if "_" in system_name else (system_name, "")
            )
            _left_lbl  = format_composition_label(_left_name)
            _right_lbl = format_composition_label(_right_name)
        _ep_fs = endpoint_label_fontsize if endpoint_label_fontsize is not None else textsize
        _xaxis_trans = ax.get_xaxis_transform()  # x in data units, y in axes-fraction units
        ax.text(0 + endpoint_label_left_dx, endpoint_label_y + endpoint_label_left_dy, _left_lbl,
                 transform=_xaxis_trans, ha="center", va="top",
                 fontsize=_ep_fs, fontweight=endpoint_label_fontweight, clip_on=False)
        ax.text(1 + endpoint_label_right_dx, endpoint_label_y + endpoint_label_right_dy, _right_lbl,
                 transform=_xaxis_trans, ha="center", va="top",
                 fontsize=_ep_fs, fontweight=endpoint_label_fontweight, clip_on=False)

    plt.tight_layout()
    return fig, ax, points, (p0, p1)


def plot_normalized_without_phases(
    all_merged, mlip, system_name,
    title=None, dodge=0.01, figsize=(5.2, 3.8),
    s=35, alpha=0.9, textsize=11,
    show_legend=True, static_tag=None, ax=None,
    mlip_marker="x", dft_marker="s",
    mlip_hollow=False, dft_hollow=False,
    mlip_alpha=None,
    dft_alpha=None,
    mlip_display=None,
    label_by="mpid",
    hull_lw=1.0,
    show_zero_line=True,
    marker_label=None,
    mlip_hull_color=None,
    dft_hull_color=None,
    hull_alpha=1.0,
    zero_line_alpha=0.35,
    phase_colors=None,
    axis_label_fontweight="normal",
):
    """
    Same as plot_normalized_with_hulls but all points share one colour
    (no per-phase colouring). Cleaner for publications.
    mlip_marker / dft_marker : any matplotlib marker string ("o", "s", "^", "D", "x", "P", …).
    mlip_hollow / dft_hollow : set True for an unfilled (outline-only) marker.
    """
    points = build_points_for_system(all_merged, mlip, system_name)
    points, p0, p1 = normalize_to_endpoints(points)

    _fp_label = mlip_display if mlip_display is not None else mlip

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs_m = [p["x"] for p in points if p.get("Erel_mlip") is not None]
    ys_m = [p["Erel_mlip"] for p in points if p.get("Erel_mlip") is not None]
    xs_d = [p["x"] for p in points if p.get("Erel_dft") is not None]
    ys_d = [p["Erel_dft"] for p in points if p.get("Erel_dft") is not None]

    _mlip_alpha = mlip_alpha if mlip_alpha is not None else alpha
    _dft_alpha  = dft_alpha  if dft_alpha  is not None else alpha
    if mlip_hollow:
        ax.scatter(xs_m, ys_m, s=s, alpha=_mlip_alpha, facecolors="none", edgecolors="tab:blue",
                   marker=mlip_marker, linewidths=1.2, label=_fp_label)
    else:
        ax.scatter(xs_m, ys_m, s=s, alpha=_mlip_alpha, c="tab:blue",
                   marker=mlip_marker, linewidths=1.2, label=_fp_label)
    if dft_hollow:
        ax.scatter(xs_d, ys_d, s=s, alpha=_dft_alpha, facecolors="none", edgecolors="tab:orange",
                   marker=dft_marker, linewidths=1.2, label="DFT")
    else:
        ax.scatter(xs_d, ys_d, s=s, alpha=_dft_alpha, c="tab:orange",
                   marker=dft_marker, label="DFT")

    mlip_xy = list(zip(xs_m, ys_m))
    dft_xy  = list(zip(xs_d, ys_d))
    if len(mlip_xy) >= 2:
        Hm = lower_hull(mlip_xy)
        ax.plot([t[0] for t in Hm], [t[1] for t in Hm], lw=hull_lw, color="tab:blue",   alpha=hull_alpha, label=f"{_fp_label} hull")
    if len(dft_xy) >= 2:
        Hd = lower_hull(dft_xy)
        ax.plot([t[0] for t in Hd], [t[1] for t in Hd], lw=hull_lw, color="tab:orange", alpha=hull_alpha, label="DFT hull")

    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                  frameon=False, ncol=4, fontsize=textsize)

    if show_zero_line:
        ax.axhline(0.0, lw=1.0, ls="--", color="black", alpha=zero_line_alpha)
    ax.set_xlabel("Atomic fraction x", fontsize=textsize, fontweight=axis_label_fontweight)
    ax.set_ylabel("$E_f$ (eV/atom)", fontsize=textsize, fontweight=axis_label_fontweight)
    if title is None:
        tag = " (static)" if static_tag else " (full)"
        display = mlip_display if mlip_display is not None else mlip
        title = f"DFT vs {display}{tag}"
    ax.set_title(title, fontsize=textsize + 2)
    plt.tight_layout()
    return fig, ax, points, (p0, p1)


def plot_hull_pair(all_merged_relax, all_merged_static, mlip, system_name,
                   plot_fn=None, figsize=(6.4, 3.3), textsize=10, dpi=350,
                   legend_ncol=4, legend_loc="lower center",
                   legend_bbox=(0.5, -0.12), legend_bottom=0.18,
                   legend_phase_ncol=3,
                   legend_marker_ncol=None,
                   legend_hull_ncol=None,
                   mlip_marker=None, dft_marker=None,
                   mlip_hollow=False, dft_hollow=False,
                   mlip_alpha=None, dft_alpha=None,
                   mlip_lw=1.0, dft_lw=1.0,
                   mlip_display=None, display_system_name=None,
                   suptitle_y=1.02, suptitle_x=0.5, suptitle_tieline=True,
                   dodge=0.01,
                   label_by="mpid",
                   hull_lw=1.0,
                   show_zero_line=True,
                   marker_label=None,
                   mlip_hull_color=None,
                   dft_hull_color=None,
                   hull_alpha=1.0,
                   zero_line_alpha=0.35,
                   phase_colors=None,
                   mlip_label_fontweight="normal",
                   mlip_label_alpha=1.0,
                   dft_label_fontweight="normal",
                   dft_label_alpha=1.0,
                   axis_label_fontweight="normal",
                   legend_fontsize=None,    # legend text size; None → same as textsize
                   subplot_left=None,       # left edge of subplots (0–1); None = matplotlib default
                   subplot_right=None,      # right edge of subplots (0–1); None = matplotlib default
                   show_endpoint_labels=False,      # show composition names at x=0 / x=1 (plot_normalized_with_hulls only)
                   endpoint_labels=None,            # (left_label, right_label) override; None → derived from system_name
                   endpoint_label_fontsize=None,    # None → falls back to textsize
                   endpoint_label_fontweight="normal",
                   endpoint_label_y=-0.10,          # base vertical position (axes fraction)
                   endpoint_label_left_dx=0.0, endpoint_label_left_dy=0.0,
                   endpoint_label_right_dx=0.0, endpoint_label_right_dy=0.0,
                   relax_title=None,       # explicit title for the left (relax/full) subplot; None → auto
                   static_title=None,      # explicit title for the right (static) subplot; None → auto
                   panel_wspace=0.12):     # horizontal gap between the two panels (matplotlib wspace units)
    """
    Plot full relaxation (left) and static (right) side-by-side with shared legend.

    plot_fn: plot_normalized_with_hulls  OR  plot_normalized_without_phases
    mlip_marker / dft_marker : forwarded to plot_fn; uses each function's default when None.
    mlip_hollow / dft_hollow : True → unfilled (outline-only) markers.
    legend_loc/legend_bbox/legend_bottom: control shared legend position.
    mlip_display: display name for the FP (e.g. "MACE"); falls back to mlip key if None.
    display_system_name: formatted system name for the figure suptitle (e.g. "GeSe₂–SiSe₂").
    show_endpoint_labels etc.: forwarded only when plot_fn is plot_normalized_with_hulls
        (plot_normalized_without_phases does not support endpoint labels).
    relax_title / static_title: override the per-subplot titles (e.g. "Full FP" / "Static FP");
        None keeps each plot_fn's default ("DFT vs {mlip} (full)"/"(static)").
    """
    if plot_fn is None:
        plot_fn = plot_normalized_without_phases

    marker_kw = {"mlip_hollow": mlip_hollow, "dft_hollow": dft_hollow,
                 "mlip_alpha": mlip_alpha, "dft_alpha": dft_alpha,
                 "mlip_lw": mlip_lw, "dft_lw": dft_lw,
                 "dodge": dodge, "label_by": label_by, "hull_lw": hull_lw,
                 "show_zero_line": show_zero_line, "marker_label": marker_label,
                 "mlip_hull_color": mlip_hull_color, "dft_hull_color": dft_hull_color,
                 "hull_alpha": hull_alpha, "zero_line_alpha": zero_line_alpha,
                 "phase_colors": phase_colors,
                 "axis_label_fontweight": axis_label_fontweight}
    if mlip_marker is not None:
        marker_kw["mlip_marker"] = mlip_marker
    if dft_marker is not None:
        marker_kw["dft_marker"] = dft_marker
    if plot_fn is plot_normalized_with_hulls:
        marker_kw.update({
            "show_endpoint_labels": show_endpoint_labels,
            "endpoint_labels": endpoint_labels,
            "endpoint_label_fontsize": endpoint_label_fontsize,
            "endpoint_label_fontweight": endpoint_label_fontweight,
            "endpoint_label_y": endpoint_label_y,
            "endpoint_label_left_dx": endpoint_label_left_dx,
            "endpoint_label_left_dy": endpoint_label_left_dy,
            "endpoint_label_right_dx": endpoint_label_right_dx,
            "endpoint_label_right_dy": endpoint_label_right_dy,
        })

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, dpi=dpi)
    plot_fn(all_merged_relax, mlip, system_name, ax=axes[0], textsize=textsize, show_legend=False,
            mlip_display=mlip_display, title=relax_title, **marker_kw)
    plot_fn(all_merged_static, mlip, system_name, static_tag=True, ax=axes[1],
            textsize=textsize, show_legend=False, mlip_display=mlip_display, title=static_title, **marker_kw)

    # Use grouped layout when per-phase handles are present, else fall back
    use_grouped = any(hasattr(_ax, "_legend_phase_handles") for _ax in axes)
    if use_grouped:
        handles, labels = _collect_legend_grouped(
            axes,
            phase_ncol=legend_phase_ncol,
            marker_ncol=legend_marker_ncol,
            hull_ncol=legend_hull_ncol,
        )
        ncol = legend_phase_ncol
    else:
        handles, labels = _collect_legend(axes)
        ncol = legend_ncol
    handler_map: dict = {}
    for _ax in axes:
        handler_map.update(getattr(_ax, "_legend_handler_map", {}))
    _leg_fs = legend_fontsize if legend_fontsize is not None else textsize
    leg = fig.legend(handles, labels, loc=legend_loc, bbox_to_anchor=legend_bbox,
                     ncol=ncol, frameon=True, fontsize=_leg_fs,
                     handler_map=handler_map or None)

    # Apply per-label fontweight / alpha to the marker-type entries (first block in grouped layout).
    # First non-blank text = MLIP label; second non-blank = DFT label (only when "separate").
    if use_grouped:
        _eff_n = (legend_marker_ncol if legend_marker_ncol is not None else legend_phase_ncol) or ncol
        _marker_texts = [t for t in leg.get_texts()[:_eff_n] if t.get_text()]
        if len(_marker_texts) >= 1:
            _marker_texts[0].set_fontweight(mlip_label_fontweight)
            _marker_texts[0].set_alpha(mlip_label_alpha)
        if len(_marker_texts) >= 2:
            _marker_texts[1].set_fontweight(dft_label_fontweight)
            _marker_texts[1].set_alpha(dft_label_alpha)

    if display_system_name is not None:
        suffix = " tie-line" if suptitle_tieline else ""
        fig.suptitle(f"{display_system_name}{suffix}", fontsize=textsize + 3, x=suptitle_x, y=suptitle_y)

    plt.tight_layout()
    _adj = {"bottom": legend_bottom, "wspace": panel_wspace}
    if subplot_left  is not None: _adj["left"]  = subplot_left
    if subplot_right is not None: _adj["right"] = subplot_right
    plt.subplots_adjust(**_adj)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# D. Energy metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mae_rmse(errs) -> tuple[float, float]:
    e = np.asarray(errs, float)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def pretty_comp_from_key(comp_key: str) -> str:
    """Try to make a human-readable composition string."""
    try:
        return str(Composition(comp_key).reduced_formula)
    except Exception:
        return comp_key


def prepare_points_for_metrics(all_merged: dict, mlip: str, system_name: str) -> list[dict]:
    """Flat list of point dicts ready for metric functions."""
    dsys = all_merged[mlip][system_name]
    pts = []
    for key, entry in dsys.items():
        dft  = entry.get("DFT_energy/atom")
        mlip_val = entry.get("MLIP_energy/atom")
        if dft is None or mlip_val is None:
            continue
        pts.append({
            "key":         key,
            "sid":         key,
            "base":        extract_base(key),
            "comp_key":    str(entry.get("composition", "")),
            "dft":         dft,
            "mlip":        mlip_val,
            "is_endpoint": ("__" in key),
            "x":           entry.get("x", 0),
        })
    return pts


def get_system_metrics(pts: list[dict]) -> dict | None:
    if not pts:
        return None
    errs = [p["mlip"] - p["dft"] for p in pts]
    mae_v, rmse_v = mae_rmse(errs)
    gp_frac,  _, _    = compute_gp_same_frac(pts)
    bb_frac, _, _, _  = compute_basebest_same_frac(pts)
    dft_min  = min(pts, key=lambda p: p["dft"])["sid"]
    mlip_min = min(pts, key=lambda p: p["mlip"])["sid"]
    return {
        "MAE":            mae_v,
        "RMSE":           rmse_v,
        "GP_Frac":        gp_frac,
        "BaseBest_Frac":  bb_frac,
        "Hull_Min_Match": dft_min == mlip_min,
        "Count":          len(pts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E. Agreement metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gp_same_frac(pts: list[dict]) -> tuple[float, int, int]:
    """
    Ground-phase agreement: per composition, does the MLIP agree with DFT
    on the lowest-energy phase?
    """
    by_comp: dict[str, list] = {}
    for p in pts:
        by_comp.setdefault(p["comp_key"], []).append(p)
    same = total = 0
    for rows in by_comp.values():
        dft_best  = min(rows, key=lambda r: r["dft"])
        mlip_best = min(rows, key=lambda r: r["mlip"])
        total += 1
        if dft_best["sid"] == mlip_best["sid"]:
            same += 1
    frac = same / total if total > 0 else float("nan")
    return frac, same, total


def compute_basebest_same_frac(pts: list[dict]) -> tuple[float, int, int, dict]:
    """
    Within-phase best agreement: per mp-id base, does the MLIP pick the same
    lowest-energy composition as DFT?
    """
    by_base: dict[str, list] = {}
    for p in pts:
        by_base.setdefault(p["base"], []).append(p)
    same = total = 0
    details: dict[str, dict] = {}
    for base, rows in sorted(by_base.items()):
        dft_best  = min(rows, key=lambda r: r["dft"])
        mlip_best = min(rows, key=lambda r: r["mlip"])
        is_same   = dft_best["sid"] == mlip_best["sid"]
        details[base] = {
            "DFT_best_id":  dft_best["sid"],
            "MLIP_best_id": mlip_best["sid"],
            "same":         bool(is_same),
        }
        total += 1
        if is_same:
            same += 1
    frac = same / total if total > 0 else float("nan")
    return frac, same, total, details


def print_gp_rankings(pts: list[dict], max_lines: int = 20):
    groups: dict[str, list] = {}
    for p in pts:
        groups.setdefault(p["comp_key"], []).append(p)
    print("\n═══ GP RANKINGS per composition (across phases) ═══")
    for comp_key, rows in sorted(groups.items()):
        by_dft  = sorted(rows, key=lambda r: r["dft"])
        by_mlip = sorted(rows, key=lambda r: r["mlip"])
        same = by_dft[0]["sid"] == by_mlip[0]["sid"]
        print(f"\n--- comp={pretty_comp_from_key(comp_key)} | N={len(rows)} | match={same} ---")
        for k, r in enumerate(by_dft[:max_lines], 1):
            print(f"  DFT  {k:2d}) {r['base']:<12s} DFT={r['dft']:+.5f}  MLIP={r['mlip']:+.5f}")
        for k, r in enumerate(by_mlip[:max_lines], 1):
            print(f"  MLIP {k:2d}) {r['base']:<12s} MLIP={r['mlip']:+.5f}  DFT={r['dft']:+.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# F. Aggregate summary table
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_system_dfs(all_merged: dict, skip_systems: set | None = None
                          ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Returns (dfs_with, dfs_without):
      dfs_with[mlip]    = DataFrame with endpoints included
      dfs_without[mlip] = DataFrame with endpoints excluded
    """
    if skip_systems is None:
        skip_systems = {"AIST_none"}
    dfs_with: dict[str, pd.DataFrame] = {}
    dfs_without: dict[str, pd.DataFrame] = {}
    for mlip in all_merged:
        rows_with, rows_without = [], []
        for system in all_merged[mlip]:
            if system in skip_systems:
                continue
            pts = prepare_points_for_metrics(all_merged, mlip, system)
            if not pts:
                continue
            r = get_system_metrics(pts)
            if r:
                rows_with.append({**r, "system": system})
            no_ep = [p for p in pts if not p["is_endpoint"]]
            r2 = get_system_metrics(no_ep)
            if r2:
                rows_without.append({**r2, "system": system})
        if rows_with:
            dfs_with[mlip]    = pd.DataFrame(rows_with)
        if rows_without:
            dfs_without[mlip] = pd.DataFrame(rows_without)
    return dfs_with, dfs_without


def create_aggregate_summary(all_dfs_dict: dict, all_merged: dict,
                              include_endpoints: bool = True,
                              skip_systems: set | None = None) -> pd.DataFrame:
    if skip_systems is None:
        skip_systems = {"AIST_none"}
    rows = []
    for mlip, system_df in all_dfs_dict.items():
        all_errors = []
        for system_name in all_merged[mlip]:
            if system_name in skip_systems:
                continue
            pts = prepare_points_for_metrics(all_merged, mlip, system_name)
            if not include_endpoints:
                pts = [p for p in pts if not p["is_endpoint"]]
            all_errors.extend([p["mlip"] - p["dft"] for p in pts])
        if not all_errors:
            continue
        arr = np.array(all_errors)
        mae_p  = float(np.mean(np.abs(arr)))
        rmse_p = float(np.sqrt(np.mean(arr ** 2)))
        rows.append({
            "MLIP":                       mlip,
            "MAE / RMSE (eV/atom)":       f"{mae_p:.4f} / {rmse_p:.4f}",
            "Ground-state agreement":     f"{system_df['GP_Frac'].mean():.1%}",
            "Within-phase min agreement": f"{system_df['BaseBest_Frac'].mean():.1%}",
            "Hull-min agreement":         f"{system_df['Hull_Min_Match'].mean():.1%}",
            "N systems":                  len(system_df),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# G. Structure comparison (RMSD)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rmsd_table(all_merged: dict, all_DFT_with_endpoints: dict,
                       skip_systems: set | None = None,
                       ltol: float = 0.5, stol: float = 0.5,
                       angle_tol: float = 10.0) -> pd.DataFrame:
    """
    Compare MLIP-relaxed structures to DFT final structures via StructureMatcher.

    Returns DataFrame with columns:
      Potential, System, Entry,
      RMSD          – pymatgen normalized value (dimensionless, divided by (V/N)^1/3)
      Max_Dist      – normalized maximum single-atom displacement (same units as RMSD)
      RMSD_Ang      – actual RMS displacement in Å  (= RMSD × (avg_vol/N)^1/3)
      Max_Dist_Ang  – actual maximum atomic displacement in Å
      norm_factor   – (avg_vol/N)^1/3 in Å used for un-normalization
      energy_error_per_atom
    """
    if skip_systems is None:
        skip_systems = {"AIST_none"}
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False,
                                ltol=ltol, stol=stol, angle_tol=angle_tol)
    rows = []
    for potential, systems in all_merged.items():
        for system_name, entries in systems.items():
            if system_name in skip_systems:
                continue
            for key, data in entries.items():
                if system_name not in all_DFT_with_endpoints:
                    continue
                if key not in all_DFT_with_endpoints[system_name]:
                    continue
                struct_mlip = data["structure"]
                struct_dft  = all_DFT_with_endpoints[system_name][key]
                dft_Epa  = data.get("DFT_energy/atom")
                mlip_Epa = data.get("MLIP_energy/atom")
                energy_err = abs(dft_Epa - mlip_Epa) if (dft_Epa and mlip_Epa) else float("nan")

                # pymatgen (scale=True) rescales both structures to the same
                # volume before matching: each lattice matrix is multiplied by
                #   ratio = (V2/V1)^(1/6)
                # so both end up at the geometric mean volume sqrt(V1*V2).
                # The normalization applied to distances is then
                #   (N / sqrt(V1*V2))^(1/3)
                # and RMSD_Ang = RMSD_normalized * (sqrt(V1*V2) / N)^(1/3).
                n_atoms     = len(struct_mlip)
                geom_vol    = (struct_mlip.volume * struct_dft.volume) ** 0.5
                norm_factor = (geom_vol / n_atoms) ** (1.0 / 3.0)

                comp = matcher.get_rms_dist(struct_mlip, struct_dft)
                rms, max_d = comp if comp is not None else (float("nan"), float("nan"))
                rows.append({
                    "Potential":             potential,
                    "System":                system_name,
                    "Entry":                 key,
                    "RMSD":                  rms,
                    "Max_Dist":              max_d,
                    "RMSD_Ang":              rms    * norm_factor,
                    "Max_Dist_Ang":          max_d  * norm_factor,
                    "norm_factor":           norm_factor,
                    "energy_error_per_atom": energy_err,
                })
    return pd.DataFrame(rows)


def rmsd_summary(df: pd.DataFrame, desired_order: list | None = None) -> pd.DataFrame:
    """Per-potential RMSD summary table.

    Deduplicates by (Potential, Entry) before computing stats so that endpoint
    structures shared across multiple systems are counted only once.

    Threshold percentages are computed over successfully mapped structures only
    (denominator = succ). Map success (%) uses total as denominator.
    Rows where succ == 0 return NaN for all threshold columns.

    Columns:
      Normalized block  – RMSD / (geom_mean_vol/N)^(1/3), dimensionless (pymatgen convention)
      Actual Å block    – RMSD_Ang = RMSD × (geom_mean_vol/N)^(1/3)
    """
    df = df.drop_duplicates(subset=["Potential", "Entry"])
    def stats(g):
        total = len(g)
        succ  = int(g["RMSD"].notna().sum())
        # threshold percentages: NaN when nothing mapped
        def pct_norm(thr):
            if succ == 0:
                return float("nan")
            return 100 * (g["RMSD"] < thr).sum() / succ
        def pct_ang(thr):
            if succ == 0:
                return float("nan")
            return 100 * (g["RMSD_Ang"] < thr).sum() / succ
        return pd.Series({
            # ── normalized (dimensionless) ─────────────────────────────────
            "Avg RMSD [norm]":      g["RMSD"].mean(),
            "Max RMSD [norm]":      g["RMSD"].max(),
            "Map success (%)":      100 * succ / total,
            "RMSD_norm < 0.01 (%)": pct_norm(0.01),
            "RMSD_norm < 0.05 (%)": pct_norm(0.05),
            "RMSD_norm < 0.10 (%)": pct_norm(0.10),
            # ── actual Å ──────────────────────────────────────────────────
            "Avg RMSD (Å)":         g["RMSD_Ang"].mean(),
            "Max RMSD (Å)":         g["RMSD_Ang"].max(),
            "Avg Max-dist (Å)":     g["Max_Dist_Ang"].mean(),
            "RMSD < 0.05 Å (%)":    pct_ang(0.05),
            "RMSD < 0.10 Å (%)":    pct_ang(0.10),
            "RMSD < 0.20 Å (%)":    pct_ang(0.20),
        })
    summary = df.groupby("Potential", group_keys=False).apply(stats)
    if desired_order:
        summary = summary.reindex([p for p in desired_order if p in summary.index])
    return summary.round(3)


# ═══════════════════════════════════════════════════════════════════════════════
# H. Ordering metrics
# ═══════════════════════════════════════════════════════════════════════════════

def get_Epa(struct_dict: dict, Etot: float) -> float:
    """Total energy / number of atoms."""
    s = Structure.from_dict(struct_dict)
    return Etot / len(s)


def ordered_index(name: str) -> int:
    m = re.search(r"ordered_(\d+)", name)
    return int(m.group(1)) if m else int(1e9)


def compute_metrics_one_group(group_dict: dict, mode: str = "relax",
                               recall_ks: tuple = (3, 10)) -> dict | None:
    """
    Compute ordering metrics for one composition group.

    Parameters
    ----------
    group_dict : {ordered_name: {relax: {...}, static: {...}}}
    mode       : "relax" or "static"
    recall_ks  : tuple of k values for recall@k
    """
    from scipy.stats import spearmanr, kendalltau

    rows = []
    for ordered, rec in group_dict.items():
        if mode not in rec:
            continue
        dft_E  = rec[mode]["DFT_final_energy"]
        mlip_E = rec[mode]["MLIP_result"]["energy"]
        struct_key = "DFT_initial_structure" if mode == "relax" else "DFT_final_structure"
        struct = rec[mode].get(struct_key)
        if struct is None:
            continue
        rows.append((ordered, get_Epa(struct, dft_E), get_Epa(struct, mlip_E)))

    n = len(rows)
    if n < 2:
        return None

    rows.sort(key=lambda x: ordered_index(x[0]))
    dft  = np.array([r[1] for r in rows])
    mlip = np.array([r[2] for r in rows])

    spear = spearmanr(dft, mlip).correlation
    kend  = kendalltau(dft, mlip, variant="b").correlation

    mismatches = 0
    total_pairs = n * (n - 1) // 2
    delta_dft = []
    for i in range(n):
        for j in range(i + 1, n):
            if (dft[i] < dft[j]) != (mlip[i] < mlip[j]):
                mismatches += 1
                delta_dft.append(abs(dft[i] - dft[j]))

    errs = mlip - dft
    mae_v, rmse_v = mae_rmse(errs)

    dft_rank  = np.argsort(dft)
    mlip_rank = np.argsort(mlip)

    out = {
        "N":                  n,
        "spearman":           float(spear) if spear is not None else float("nan"),
        "kendall":            float(kend)  if kend  is not None else float("nan"),
        "rank_error_frac":    mismatches / total_pairs if total_pairs > 0 else float("nan"),
        "MAE":                mae_v,
        "RMSE":               rmse_v,
        "DeltaE_DFT_mean":    float(np.mean(delta_dft)) if delta_dft else 0.0,
        "DeltaE_DFT_max":     float(np.max(delta_dft))  if delta_dft else 0.0,
        "top1_acc":           float(int(dft_rank[0] == mlip_rank[0])),
    }
    for k in recall_ks:
        if n >= k:
            out[f"recall@{k}"] = float(len(set(dft_rank[:k]) & set(mlip_rank[:k])) / k)

    return out


def summarize_one_potential(pot_groups_dict: dict, recall_ks: tuple = (3, 10)) -> tuple[dict, dict]:
    """Macro-average ordering metrics over all composition groups for one MLIP."""
    per_relax: dict[str, dict]  = {}
    per_static: dict[str, dict] = {}
    for parent, group_dict in pot_groups_dict.items():
        r = compute_metrics_one_group(group_dict, mode="relax",  recall_ks=recall_ks)
        s = compute_metrics_one_group(group_dict, mode="static", recall_ks=recall_ks)
        if r: per_relax[parent]  = r
        if s: per_static[parent] = s

    def macro(d, key):
        vals = [v.get(key, float("nan")) for v in d.values()]
        vals = [x for x in vals if x == x]
        return float(np.mean(vals)) if vals else float("nan")

    keys = ["top1_acc", "recall@3", "recall@10",
            "spearman", "kendall", "rank_error_frac",
            "MAE", "RMSE", "DeltaE_DFT_mean", "DeltaE_DFT_max"]

    def global_errs(mode, struct_key):
        errs = []
        for parent, group_dict in pot_groups_dict.items():
            for ordered, rec in group_dict.items():
                if mode not in rec:
                    continue
                dft_E  = rec[mode]["DFT_final_energy"]
                mlip_E = rec[mode]["MLIP_result"]["energy"]
                s = rec[mode].get(struct_key)
                if s:
                    errs.append(get_Epa(s, mlip_E) - get_Epa(s, dft_E))
        return errs

    def build_summary(per, mode, struct_key):
        d = {f"avg_{k}": macro(per, k) for k in keys}
        d["N_groups"] = len(per)
        ge = global_errs(mode, struct_key)
        d["global_MAE"],  d["global_RMSE"] = mae_rmse(ge) if ge else (float("nan"), float("nan"))
        d["N_points"] = len(ge)
        return d

    return (
        build_summary(per_relax,  "relax",  "DFT_initial_structure"),
        build_summary(per_static, "static", "DFT_final_structure"),
    )


def build_ordering_summary(merged: dict, recall_ks: tuple = (3, 10),
                            desired_order: list | None = None
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run summarize_one_potential for every MLIP in merged.

    Returns (df_relax, df_static) indexed by potential name.
    """
    rows_r, rows_s = [], []
    for pot_name, pot_groups in merged.items():
        if not isinstance(pot_groups, dict):
            continue
        s_r, s_s = summarize_one_potential(pot_groups, recall_ks=recall_ks)
        rows_r.append({"potential": pot_name, **s_r})
        rows_s.append({"potential": pot_name, **s_s})

    df_r = pd.DataFrame(rows_r).set_index("potential").sort_index()
    df_s = pd.DataFrame(rows_s).set_index("potential").sort_index()

    if desired_order:
        df_r = df_r.reindex([p for p in desired_order if p in df_r.index])
        df_s = df_s.reindex([p for p in desired_order if p in df_s.index])

    return df_r, df_s


# ═══════════════════════════════════════════════════════════════════════════════
# I. Standardized data model: dft_hull / fp_hull / dft_ordering / fp_ordering
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the all_mlips_clean_*/endpoints_*/ordering_merged pipeline above with
# four explicit, provenance-preserving dictionaries. Candidate identity is
# stable and shared between DFT and every FP: a composition string for interior
# hull structures, an mp-id for endpoints, and the original `ordered_name` for
# ordering-benchmark configurations. No silent structure fallback: a missing
# FP-relaxed structure is status="missing", never replaced by a DFT structure.

_ELEM_TOKEN = re.compile(r"^[A-Z][a-z]?\d[+-]\d+$")


def phase_id_of(composition: str) -> str:
    """
    Phase/prototype identifier: every underscore-delimited token in a
    composition string before the first element-valence-count token
    (e.g. 'Ge4+1'). Usually an MP-id ('mp-10074'), but a real minority of
    compositions in this dataset use a space-group-based prototype label
    instead (e.g. 'r3m_SnTe') because they were generated from a manually
    specified prototype with no associated MP entry. Do not assume the
    'mp-NNNNN' pattern unconditionally -- it does not hold for ~4% of
    compositions in the real data.
    """
    parts = composition.split("_")
    prefix = []
    for p in parts:
        if _ELEM_TOKEN.match(p):
            break
        prefix.append(p)
    if not prefix:
        raise ValueError(f"composition {composition!r}: no phase-id prefix found")
    return "_".join(prefix)


def _extract_hull_records_format_a(hull_raw: dict) -> dict:
    """FORMAT A (flat) hull JSON -> {(system, composition): record}."""
    out = {}
    for sid, e in hull_raw.items():
        sk = e.get("source_keys", {})
        system, comp = sk.get("system"), sk.get("composition")
        ordered_name, tie_line = sk.get("ordered_name"), sk.get("tie_line")
        if system is None or comp is None:
            parts = sid.split("||")
            if len(parts) >= 4:
                tie_line, system, comp, ordered_name = parts[0], parts[1], parts[2], parts[3]
        mlip_res = e.get("MLIP_result", {})
        out[(system, comp)] = {
            "tie_line": tie_line, "ordered_name": ordered_name,
            "dft_energy": e.get("DFT_final_energy"),
            "dft_initial_structure": e.get("DFT_initial_structure"),
            "dft_final_structure": e.get("DFT_final_structure"),
            "fp_energy": mlip_res.get("energy"),
            "fp_relaxed_structure": mlip_res.get("relaxed_structure"),
        }
    return out


def _extract_hull_records_format_b(hull_raw: dict, want_fp_structure: bool) -> dict:
    """FORMAT B (nested) hull JSON section -> {(system, composition): record}.

    FORMAT B never stores DFT structures inline (only dft_E0/dft_Ef energies +
    forces) -- dft_initial_structure/dft_final_structure are always None here;
    the DFT reference structure must come from a FORMAT A source.
    """
    out = {}
    for vname, vdata in hull_raw.items():
        for system, sdata in vdata.items():
            for comp, cdata in sdata.items():
                for struct_name, sinfo in cdata.items():
                    mlip_res = sinfo.get("MLIP_result", {})
                    dft_final = sinfo.get("DFT_final_structure_data", {})
                    out[(system, comp)] = {
                        "tie_line": vname, "ordered_name": struct_name,
                        "dft_energy": dft_final.get("dft_Ef"),
                        "dft_initial_structure": None,
                        "dft_final_structure": None,
                        "fp_energy": mlip_res.get("energy"),
                        "fp_relaxed_structure": mlip_res.get("relaxed_structure") if want_fp_structure else None,
                    }
    return out


def _extract_endpoints_format_a(ep_raw: dict) -> dict:
    """FORMAT A endpoints JSON -> {sid: record}."""
    out = {}
    for eid, e in ep_raw.items():
        mlip_res = e.get("MLIP_result", {})
        sid = e.get("entry_id", eid)
        if not sid.startswith("mp-"):
            sid = f"mp-{sid}"
        sid = re.sub(r"-(GGA|LDA|SCAN|r2SCAN|PBE|PBEsol)(\+U)?$", "", sid)
        out[sid] = {
            "endpoint_id": eid,
            "dft_energy": e.get("DFT_energy"),
            "dft_structure": e.get("DFT_structure"),
            "fp_energy": mlip_res.get("energy"),
            "fp_relaxed_structure": mlip_res.get("relaxed_structure"),
        }
    return out


def _extract_endpoints_format_b(ep_raw: dict, structure_key: str = "mp_structure") -> dict:
    """FORMAT B endpoints JSON section -> {sid: record}."""
    out = {}
    for id_key, d in ep_raw.items():
        mlip_res = d.get("MLIP_result", {})
        sid = "mp-" + id_key.split("_")[-1]
        out[sid] = {
            "endpoint_id": id_key,
            "dft_energy": d.get("DFT_final_energy"),
            "dft_structure": d.get(structure_key),
            "fp_energy": mlip_res.get("energy"),
            "fp_relaxed_structure": mlip_res.get("relaxed_structure"),
        }
    return out


def build_dft_hull(hull_records_ref: dict, ep_records_ref: dict,
                    skip_systems: set | None = None) -> tuple[dict, list[str], dict]:
    """
    Build dft_hull[system][candidate_id] from ONE canonical DFT-reference
    source (hull_records_ref, ep_records_ref -- typically the FORMAT A "mace"
    extraction, since it is the only source carrying full DFT structures).

    Endpoint routing replicates the existing (indirect) two-step logic
    exactly, per "do not change endpoint normalization":
      step 1: an endpoint with mp-id `sid` is only a CANDIDATE for systems
        where some INTERIOR composition's phase-id prefix
        (comp.split("_")[0], NOT the fuller phase_id_of) equals that sid.
      step 2: among candidate systems, keep only if x<=0.05 or x>=0.95
        (loose attach), and a system counts as having a left/right endpoint
        only if some attached candidate has x<1e-5 or x>1-1e-5 (tight, "true
        corner" check) -- otherwise the system is excluded entirely, exactly
        as in the current merge_endpoints_into_hull + missing-endpoint check.

    Returns (dft_hull, final_systems, diagnostics).
    """
    if skip_systems is None:
        skip_systems = {"AIST_none"}

    system_comps: dict[str, set] = {}
    for (system, comp) in hull_records_ref:
        system_comps.setdefault(system, set()).add(comp)

    sid_to_candidate_systems: dict[str, set] = {}
    for (system, comp) in hull_records_ref:
        sid_to_candidate_systems.setdefault(comp.split("_")[0], set()).add(system)

    all_systems_raw = sorted(s for s in system_comps if s not in skip_systems)

    system_endpoints: dict[str, dict[str, list]] = {s: {"left": [], "right": []} for s in all_systems_raw}
    for sid, rec in ep_records_ref.items():
        if rec["dft_structure"] is None:
            continue
        for system in sid_to_candidate_systems.get(sid, ()):
            if system not in system_endpoints:
                continue
            try:
                frA, frB = parse_endmembers(system)
                struct = Structure.from_dict(rec["dft_structure"])
                frC, _ = structure_frac_dict(struct)
                x = compute_x_projection(frC, frA, frB)
            except Exception:
                x = None
            if x is None:
                continue
            if x <= 0.05:
                system_endpoints[system]["left"].append((sid, x))
            elif x >= 0.95:
                system_endpoints[system]["right"].append((sid, x))

    def has_true_corner(pairs, target):
        return any(abs(x - target) < 1e-5 for _, x in pairs)

    missing_endpoint_systems = [
        s for s in all_systems_raw
        if not (has_true_corner(system_endpoints[s]["left"], 0.0)
                and has_true_corner(system_endpoints[s]["right"], 1.0))
    ]
    final_systems = [s for s in all_systems_raw if s not in missing_endpoint_systems]

    dft_hull: dict[str, dict] = {}
    unique_endpoint_sids: set = set()
    for system in final_systems:
        dft_hull[system] = {}
        for comp in sorted(system_comps[system]):
            rec = hull_records_ref[(system, comp)]
            if rec["dft_final_structure"] is None:
                raise ValueError(
                    f"DFT reference has no structure for required candidate "
                    f"({system!r}, {comp!r}) -- the reference source must carry structures."
                )
            struct = Structure.from_dict(rec["dft_final_structure"])
            n_atoms = len(struct.sites)
            frA, frB = parse_endmembers(system)
            frC, _ = structure_frac_dict(struct)
            x = compute_x_projection(frC, frA, frB)
            dft_hull[system][comp] = {
                "candidate_id": comp,
                "source_keys": {"tie_line": rec["tie_line"], "system": system,
                                 "composition": comp, "ordered_name": rec["ordered_name"]},
                "role": "interior",
                "phase_id": phase_id_of(comp),
                "composition": comp,
                "ordered_name": rec["ordered_name"],
                "x": x,
                "n_atoms": n_atoms,
                "energy_total": rec["dft_energy"],
                "energy_per_atom": rec["dft_energy"] / n_atoms,
                "initial_structure": rec["dft_initial_structure"],
                "relaxed_structure": rec["dft_final_structure"],
            }

        for side, target in [("left", 0.0), ("right", 1.0)]:
            for sid, x in system_endpoints[system][side]:
                if abs(x - target) >= 1e-5:
                    continue
                rec = ep_records_ref[sid]
                struct = Structure.from_dict(rec["dft_structure"])
                n_atoms = len(struct.sites)
                dft_hull[system].setdefault(sid, {
                    "candidate_id": sid,
                    "source_keys": {"endpoint_id": rec["endpoint_id"]},
                    "role": "endpoint",
                    "endpoint_side": side,
                    "phase_id": sid,
                    "composition": str(struct.composition),
                    "x": target,
                    "n_atoms": n_atoms,
                    "energy_total": rec["dft_energy"],
                    "energy_per_atom": rec["dft_energy"] / n_atoms,
                    "initial_structure": None,
                    "relaxed_structure": rec["dft_structure"],
                })
                unique_endpoint_sids.add(sid)

    n_interior = sum(1 for s in dft_hull for cid, c in dft_hull[s].items() if c["role"] == "interior")
    diagnostics = {
        "n_systems_raw": len(all_systems_raw),
        "missing_endpoint_systems": missing_endpoint_systems,
        "n_final_systems": len(final_systems),
        "n_interior": n_interior,
        "n_unique_endpoints": len(unique_endpoint_sids),
        "n_total_candidates": n_interior + len(unique_endpoint_sids),
    }
    return dft_hull, final_systems, diagnostics


def build_fp_hull(dft_hull: dict, final_systems: list[str],
                   hull_relax_records: dict, hull_static_records: dict,
                   ep_relax_records: dict, ep_static_records: dict) -> dict:
    """
    Build fp_hull["relax"/"static"][system][candidate_id] for ONE FP from its
    own extracted records, matched against dft_hull's candidate universe.

    status is "success" or "missing" only. A missing FP-relaxed structure is
    NEVER replaced by the DFT initial/final structure -- if MLIP_result has no
    usable energy (or, for relax, no relaxed_structure), status="missing" and
    no energy/structure fields are populated. Static evaluations are always
    scored on the canonical DFT-relaxed structure (dft_hull's own
    relaxed_structure); an FP-relaxed structure is never used as "the"
    static-evaluation structure.
    """
    fp_hull = {"relax": {}, "static": {}}
    for mode, records, ep_records in [
        ("relax", hull_relax_records, ep_relax_records),
        ("static", hull_static_records, ep_static_records),
    ]:
        fp_hull[mode] = {system: {} for system in final_systems}
        for system in final_systems:
            for candidate_id, dft_entry in dft_hull[system].items():
                if dft_entry["role"] == "interior":
                    rec = records.get((system, candidate_id))
                    fp_energy = rec["fp_energy"] if rec else None
                    fp_struct = rec["fp_relaxed_structure"] if (rec and mode == "relax") else None
                else:
                    rec = ep_records.get(candidate_id)
                    fp_energy = rec["fp_energy"] if rec else None
                    fp_struct = rec["fp_relaxed_structure"] if (rec and mode == "relax") else None

                if rec is None or fp_energy is None or (mode == "relax" and fp_struct is None):
                    fp_hull[mode][system][candidate_id] = {"status": "missing"}
                    continue

                n_atoms = dft_entry["n_atoms"]
                entry = {
                    "status": "success",
                    "energy_total": fp_energy,
                    "energy_per_atom": fp_energy / n_atoms,
                }
                if mode == "relax":
                    entry["relaxed_structure"] = fp_struct
                fp_hull[mode][system][candidate_id] = entry
    return fp_hull


def build_dft_ordering(ordering_raw_ref: dict, case_alias_compositions: set[str] | None = None) -> dict:
    """
    Build dft_ordering[group_name] from ONE canonical DFT-reference ordering
    source (the FORMAT A "mace" GROUPED20 file). group_name is the composition
    string ("existing composition-and-phase group name").

    Deduplicates raw group keys that share a composition but differ only by a
    filesystem-casing artifact in the tie_line component (confirmed
    byte-identical for the known 8-pair set in this dataset -- see validation
    record). Any OTHER duplicate composition (not in case_alias_compositions)
    raises, per "stop and report the collisions rather than silently
    overwriting them or inventing a more complicated key."
    """
    case_alias_compositions = case_alias_compositions or set()

    comp_to_rawkeys: dict[str, list[str]] = {}
    for k in ordering_raw_ref:
        parts = k.split("||")
        if len(parts) != 3:
            raise ValueError(f"unexpected ordering group key shape: {k!r}")
        comp = parts[2]
        comp_to_rawkeys.setdefault(comp, []).append(k)

    unexpected = {c: ks for c, ks in comp_to_rawkeys.items()
                  if len(ks) > 1 and c not in case_alias_compositions}
    if unexpected:
        raise ValueError(
            "Unexpected duplicate ordering-group composition keys (not in the "
            f"confirmed case-alias list): {unexpected}. Stopping rather than "
            "silently merging or inventing a more complicated key."
        )
    confirmed_but_absent = case_alias_compositions - {c for c, ks in comp_to_rawkeys.items() if len(ks) > 1}
    if confirmed_but_absent:
        raise ValueError(
            f"Confirmed case-alias compositions not found as duplicates in the "
            f"reference ordering data: {confirmed_but_absent}"
        )

    dft_ordering = {}
    for comp, rawkeys in comp_to_rawkeys.items():
        canonical_key = sorted(rawkeys)[0]
        tie_line, system, _ = canonical_key.split("||")
        group_dict = ordering_raw_ref[canonical_key]
        if len(group_dict) != 20:
            raise ValueError(f"group {comp!r} does not have exactly 20 orderings ({len(group_dict)})")

        orderings = {}
        for ordered_name, rec in group_dict.items():
            relax = rec.get("relax", {})
            dft_E = relax.get("DFT_final_energy")
            struct = relax.get("DFT_initial_structure")
            n_atoms = len(struct["sites"]) if struct else None
            if dft_E is None or n_atoms is None:
                raise ValueError(f"group {comp!r} ordering {ordered_name!r}: missing required DFT reference data")
            orderings[ordered_name] = {
                "n_atoms": n_atoms,
                "energy_total": dft_E,
                "energy_per_atom": dft_E / n_atoms,
                "initial_structure": relax.get("DFT_initial_structure"),
                "relaxed_structure": relax.get("DFT_final_structure"),
            }

        dft_ordering[comp] = {
            "system": system,
            "phase_id": phase_id_of(comp),
            "composition": comp,
            "source_group_keys": sorted(rawkeys),
            "orderings": orderings,
        }
    return dft_ordering


def build_fp_ordering(dft_ordering: dict, ordering_raw_fp: dict) -> dict:
    """
    Build fp_ordering["relax"/"static"][group_name][ordered_name] for ONE FP.

    Requires the DFT group to have exactly 20 configurations (already
    enforced by build_dft_ordering); does NOT require the FP side to be
    complete -- incomplete FP groups get per-ordering status="missing" and
    must be reported explicitly by validation, not silently computed on as if
    complete.
    """
    comp_to_rawkeys_fp: dict[str, list[str]] = {}
    for k in ordering_raw_fp:
        parts = k.split("||")
        if len(parts) != 3:
            continue
        comp_to_rawkeys_fp.setdefault(parts[2], []).append(k)

    fp_ordering = {"relax": {}, "static": {}}
    for mode in ("relax", "static"):
        fp_ordering[mode] = {}
        for comp, dft_group in dft_ordering.items():
            fp_ordering[mode][comp] = {}
            rawkeys = comp_to_rawkeys_fp.get(comp)
            group_dict = ordering_raw_fp[sorted(rawkeys)[0]] if rawkeys else {}
            for ordered_name in dft_group["orderings"]:
                rec = group_dict.get(ordered_name, {}).get(mode)
                n_atoms = dft_group["orderings"][ordered_name]["n_atoms"]
                if rec is None or rec.get("MLIP_result", {}).get("energy") is None:
                    fp_ordering[mode][comp][ordered_name] = {"status": "missing"}
                    continue
                e = rec["MLIP_result"]["energy"]
                fp_ordering[mode][comp][ordered_name] = {
                    "status": "success",
                    "energy_total": e,
                    "energy_per_atom": e / n_atoms,
                }
    return fp_ordering


# ═══════════════════════════════════════════════════════════════════════════════
# J. Hull metrics: pooled ground-state agreement, within-phase / global
#    hull-minimum agreement, all with explicit numerator/denominator reporting.
# ═══════════════════════════════════════════════════════════════════════════════

def _hull_energy_pairs(dft_hull: dict, fp_hull: dict, mode: str) -> dict:
    """{system: {candidate_id: (dft_Epa, fp_Epa)}} for successfully-scored candidates."""
    out = {}
    for system, candidates in dft_hull.items():
        out[system] = {}
        for cid, dft_entry in candidates.items():
            fp_entry = fp_hull[mode][system].get(cid)
            if fp_entry is None or fp_entry.get("status") != "success":
                continue
            out[system][cid] = (dft_entry["energy_per_atom"], fp_entry["energy_per_atom"])
    return out


def ground_state_agreement_pooled(dft_hull: dict, fp_hull: dict, mode: str) -> dict:
    """
    Ground-state agreement, POOLED (not macro-averaged): for every composition
    instance (same chemical formula, i.e. same set of competing phases) within
    a system, does the FP pick the same lowest-energy phase as DFT? The
    reported fraction is (matching instances) / (valid instances) pooled
    across every system directly -- each composition instance is weighted
    equally, not each system.

    A "composition instance" requires >=2 competing phases; single-phase
    compositions provide no ground-state choice and are excluded (reported
    separately, not silently dropped).
    """
    pairs = _hull_energy_pairs(dft_hull, fp_hull, mode)
    groups: dict[tuple, list] = {}
    for system, cand_pairs in pairs.items():
        for cid, (dft_e, fp_e) in cand_pairs.items():
            dft_entry = dft_hull[system][cid]
            struct = Structure.from_dict(dft_entry["relaxed_structure"])
            comp_key = str(struct.composition)
            groups.setdefault((system, comp_key), []).append((cid, dft_e, fp_e))

    numerator = denominator = excluded_single_phase = 0
    for (system, comp_key), rows in groups.items():
        if len(rows) < 2:
            excluded_single_phase += 1
            continue
        dft_best = min(rows, key=lambda r: r[1])
        fp_best = min(rows, key=lambda r: r[2])
        denominator += 1
        if dft_best[0] == fp_best[0]:
            numerator += 1

    return {
        "fraction": (numerator / denominator) if denominator else float("nan"),
        "numerator": numerator,
        "denominator": denominator,
        "excluded_single_phase_instances": excluded_single_phase,
    }


def within_phase_hull_min_agreement(dft_hull: dict, fp_hull: dict, mode: str) -> dict:
    """
    Within-phase hull-minimum agreement: for each phase (phase_id) within a
    tie-line system, does the FP identify the same minimum-(normalized-)energy
    composition as DFT? Uses normalized (tie-line-relative) energies, per the
    existing normalize_to_endpoints method -- unchanged.

    Phases with only one composition present provide no within-phase choice
    and are reported separately (excluded from the fraction, not silently
    dropped or treated as automatic agreement).
    """
    numerator = denominator = excluded_single_composition = 0
    undefined_normalization = 0
    for system in dft_hull:
        try:
            raw_pts = build_points_for_system({"__fp__": {system: {
                cid: {"structure": Structure.from_dict(e["relaxed_structure"]),
                      "MLIP_energy/atom": (fp_hull[mode][system].get(cid) or {}).get("energy_per_atom"),
                      "DFT_energy/atom": e["energy_per_atom"]}
                for cid, e in dft_hull[system].items()
                if (fp_hull[mode][system].get(cid) or {}).get("status") == "success"
            }}}, "__fp__", system)
        except Exception:
            undefined_normalization += 1
            continue
        try:
            points, _, _ = normalize_to_endpoints(raw_pts)
        except Exception:
            undefined_normalization += 1
            continue
        valid = [p for p in points if p.get("Erel_dft") is not None and p.get("Erel_mlip") is not None]

        # Phase grouping uses the explicit phase_id field stored on each
        # dft_hull candidate, not a candidate_id string convention -- so an
        # external dataset's candidate_id does not need to embed phase_id.
        by_phase: dict[str, list] = {}
        for p in valid:
            phase = dft_hull[system][p["key"]]["phase_id"]
            by_phase.setdefault(phase, []).append(p)

        for phase, pts in by_phase.items():
            if len(pts) < 2:
                excluded_single_composition += 1
                continue
            dft_best = min(pts, key=lambda p: p["Erel_dft"])
            fp_best = min(pts, key=lambda p: p["Erel_mlip"])
            denominator += 1
            if dft_best["key"] == fp_best["key"]:
                numerator += 1

    return {
        "fraction": (numerator / denominator) if denominator else float("nan"),
        "numerator": numerator,
        "denominator": denominator,
        "excluded_single_composition_phases": excluded_single_composition,
        "undefined_normalization_systems": undefined_normalization,
    }


def global_hull_min_agreement(dft_hull: dict, fp_hull: dict, mode: str) -> dict:
    """
    Global hull-minimum agreement: for each tie-line system, does the FP
    identify the same global convex-hull minimum (across ALL phases and
    compositions) as DFT? Uses normalized energies, unchanged. Reports the
    DFT- and FP-selected candidate_id for every system, not just the pooled
    fraction.
    """
    numerator = denominator = 0
    per_system = {}
    for system in dft_hull:
        try:
            raw_pts = build_points_for_system({"__fp__": {system: {
                cid: {"structure": Structure.from_dict(e["relaxed_structure"]),
                      "MLIP_energy/atom": (fp_hull[mode][system].get(cid) or {}).get("energy_per_atom"),
                      "DFT_energy/atom": e["energy_per_atom"]}
                for cid, e in dft_hull[system].items()
                if (fp_hull[mode][system].get(cid) or {}).get("status") == "success"
            }}}, "__fp__", system)
            points, _, _ = normalize_to_endpoints(raw_pts)
        except Exception:
            continue
        valid = [p for p in points if p.get("Erel_dft") is not None and p.get("Erel_mlip") is not None]
        if not valid:
            continue
        dft_min = min(valid, key=lambda p: p["Erel_dft"])
        fp_min = min(valid, key=lambda p: p["Erel_mlip"])
        match = dft_min["key"] == fp_min["key"]
        denominator += 1
        if match:
            numerator += 1
        per_system[system] = {"dft_candidate_id": dft_min["key"], "fp_candidate_id": fp_min["key"], "match": match}

    return {
        "fraction": (numerator / denominator) if denominator else float("nan"),
        "numerator": numerator,
        "denominator": denominator,
        "per_system": per_system,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# K. Structure comparison (RMSD) with corrected failure/mapping denominators
# ═══════════════════════════════════════════════════════════════════════════════

def _unique_candidates(dft_hull: dict) -> list[tuple]:
    """
    All intended unique convex-hull candidates, deduplicated by stable
    identity: interior candidates once per (system, candidate_id) since a
    composition is scoped to its own system; endpoints once per candidate_id
    globally (a shared endpoint appearing in multiple systems' views is the
    same physical structure and must count once).
    Returns list of (system, candidate_id, dft_entry) -- for a deduplicated
    endpoint, `system` is the FIRST system (by sorted order) it appears in,
    used only to fetch a representative dft_entry.
    """
    seen_endpoints: set = set()
    out = []
    for system in sorted(dft_hull):
        for cid, dft_entry in sorted(dft_hull[system].items()):
            if dft_entry["role"] == "endpoint":
                if cid in seen_endpoints:
                    continue
                seen_endpoints.add(cid)
            out.append((system, cid, dft_entry))
    return out


def compute_structure_rmsd(dft_hull: dict, fp_hull: dict, mode: str,
                     ltol: float = 0.5, stol: float = 0.5, angle_tol: float = 10.0) -> pd.DataFrame:
    """
    Compare fp_hull[mode][system][candidate_id]['relaxed_structure'] against
    dft_hull[system][candidate_id]['relaxed_structure'] via StructureMatcher,
    for every unique intended candidate (deduplicated endpoints included).

    Distinguishes:
      - calculation_failure: fp_hull status != "success" (no FP structure to
        compare at all -- failed relaxation / missing FP-relaxed structure /
        missing result). RMSD is NaN, mapped=False.
      - mapping_failure: FP structure exists but StructureMatcher.get_rms_dist
        returns None (structures too dissimilar to map). RMSD is NaN,
        mapped=False.
      - success: mapped=True, RMSD/Max_Dist populated.

    Map success (%) denominator = ALL unique intended candidates (both
    failure types count against it). Mean/max RMSD and threshold fractions
    are computed only over mapped=True rows. Same StructureMatcher
    tolerances and Å conversion as before, unchanged.
    """
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False,
                                ltol=ltol, stol=stol, angle_tol=angle_tol)
    rows = []
    for system, cid, dft_entry in _unique_candidates(dft_hull):
        fp_entry = fp_hull[mode][system].get(cid)
        struct_dft = Structure.from_dict(dft_entry["relaxed_structure"])

        if fp_entry is None or fp_entry.get("status") != "success" or "relaxed_structure" not in fp_entry:
            rows.append({
                "System": system, "Entry": cid, "role": dft_entry["role"],
                "failure_type": "calculation_failure",
                "mapped": False, "RMSD": float("nan"), "Max_Dist": float("nan"),
                "RMSD_Ang": float("nan"), "Max_Dist_Ang": float("nan"),
                "norm_factor": float("nan"), "energy_error_per_atom": float("nan"),
            })
            continue

        struct_fp = Structure.from_dict(fp_entry["relaxed_structure"])
        n_atoms = len(struct_fp)
        geom_vol = (struct_fp.volume * struct_dft.volume) ** 0.5
        norm_factor = (geom_vol / n_atoms) ** (1.0 / 3.0)
        comp = matcher.get_rms_dist(struct_fp, struct_dft)

        dft_Epa = dft_entry["energy_per_atom"]
        fp_Epa = fp_entry["energy_per_atom"]
        energy_err = abs(dft_Epa - fp_Epa)

        if comp is None:
            rows.append({
                "System": system, "Entry": cid, "role": dft_entry["role"],
                "failure_type": "mapping_failure",
                "mapped": False, "RMSD": float("nan"), "Max_Dist": float("nan"),
                "RMSD_Ang": float("nan"), "Max_Dist_Ang": float("nan"),
                "norm_factor": norm_factor, "energy_error_per_atom": energy_err,
            })
            continue

        rms, max_d = comp
        rows.append({
            "System": system, "Entry": cid, "role": dft_entry["role"],
            "failure_type": None,
            "mapped": True, "RMSD": rms, "Max_Dist": max_d,
            "RMSD_Ang": rms * norm_factor, "Max_Dist_Ang": max_d * norm_factor,
            "norm_factor": norm_factor, "energy_error_per_atom": energy_err,
        })
    return pd.DataFrame(rows)


def summarize_structure_rmsd(df: pd.DataFrame) -> dict:
    """
    Summary stats from compute_structure_rmsd's output (single FP/mode already).

    Map success denominator = len(df) (every unique intended candidate).
    Mean/max RMSD and threshold fractions computed only over mapped rows.
    Reports calculation_failure and mapping_failure counts separately.
    """
    total = len(df)
    mapped = df[df["mapped"]]
    n_mapped = len(mapped)
    n_calc_fail = int((df["failure_type"] == "calculation_failure").sum())
    n_map_fail = int((df["failure_type"] == "mapping_failure").sum())

    def pct_ang(thr):
        return 100 * (mapped["RMSD_Ang"] < thr).sum() / n_mapped if n_mapped else float("nan")

    return {
        "n_total_candidates": total,
        "n_mapped": n_mapped,
        "n_calculation_failure": n_calc_fail,
        "n_mapping_failure": n_map_fail,
        "map_success_pct": 100 * n_mapped / total if total else float("nan"),
        "avg_rmsd_ang": mapped["RMSD_Ang"].mean() if n_mapped else float("nan"),
        "max_rmsd_ang": mapped["RMSD_Ang"].max() if n_mapped else float("nan"),
        "rmsd_lt_0.05_pct": pct_ang(0.05),
        "rmsd_lt_0.10_pct": pct_ang(0.10),
        "rmsd_lt_0.20_pct": pct_ang(0.20),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# L. Ordering metrics: manuscript-exact ranking-error formula, complete-group
#    requirement, per-group metrics macro-averaged, used identically by both the
#    main ordering summary and the single-group demonstration plot.
# ═══════════════════════════════════════════════════════════════════════════════

def is_misranked_pair(e_dft_i: float, e_dft_j: float, e_fp_i: float, e_fp_j: float) -> bool:
    """
    Manuscript Eq. 6 condition exactly:
        (E_dft_i - E_dft_j) * (E_fp_i - E_fp_j) <= 0
    A tie on either side (DFT tie, FP tie, or both) counts as misranked, since
    the product is exactly zero -- this is a deliberate, explicit convention
    (see conversation record / _test_is_misranked_pair for the 5 confirmed
    cases), not an accidental artifact of a boolean-< comparison.
    """
    return (e_dft_i - e_dft_j) * (e_fp_i - e_fp_j) <= 0


def _test_is_misranked_pair():
    """Small tests: correct pair, reversed pair, DFT tie, FP tie, both tied."""
    cases = [
        ("correct pair",  0.0, 1.0, 0.0, 1.0, False),
        ("reversed pair", 0.0, 1.0, 1.0, 0.0, True),
        ("DFT tie",       0.5, 0.5, 0.0, 1.0, True),
        ("FP tie",        0.0, 1.0, 0.5, 0.5, True),
        ("both tied",     0.5, 0.5, 0.5, 0.5, True),
    ]
    results = []
    for name, edi, edj, efi, efj, expected in cases:
        got = is_misranked_pair(edi, edj, efi, efj)
        results.append((name, expected, got, got == expected))
    all_pass = all(r[3] for r in results)
    return all_pass, results


def compute_ordering_group_metrics(dft_group: dict, fp_group: dict, mode: str,
                                       recall_ks: tuple = (3, 10)) -> dict | None:
    """
    Ordering metrics for ONE group, ONE FP, ONE mode. Requires the DFT group
    to have exactly 20 configurations (enforced by build_dft_ordering) AND
    the FP group to have a successful energy for every one of those same 20
    ordered_names -- an incomplete FP group returns None (report explicitly,
    do not silently compute manuscript metrics on a partial group as if it
    were complete).
    """
    from itertools import combinations
    from scipy.stats import spearmanr

    ordered_names = sorted(dft_group["orderings"].keys())
    if len(ordered_names) != 20:
        raise ValueError(f"DFT group has {len(ordered_names)} orderings, expected 20")

    missing = [n for n in ordered_names if fp_group.get(n, {}).get("status") != "success"]
    if missing:
        return {"complete": False, "missing_ordered_names": missing}

    dft_e = np.array([dft_group["orderings"][n]["energy_per_atom"] for n in ordered_names])
    fp_e = np.array([fp_group[n]["energy_per_atom"] for n in ordered_names])
    n = len(ordered_names)

    dft_rank = np.argsort(dft_e)
    fp_rank = np.argsort(fp_e)
    top1_acc = int(dft_rank[0] == fp_rank[0])

    recalls = {}
    for k in recall_ks:
        if n >= k:
            recalls[f"recall@{k}"] = len(set(dft_rank[:k]) & set(fp_rank[:k])) / k

    spearman = spearmanr(dft_e, fp_e).correlation

    n_misranked = 0
    total_pairs = n * (n - 1) // 2
    delta_dft_misranked = []
    for i, j in combinations(range(n), 2):
        if is_misranked_pair(dft_e[i], dft_e[j], fp_e[i], fp_e[j]):
            n_misranked += 1
            delta_dft_misranked.append(abs(dft_e[i] - dft_e[j]))

    errs = fp_e - dft_e
    mae_v, rmse_v = mae_rmse(errs)

    return {
        "complete": True,
        "N": n,
        "top1_acc": float(top1_acc),
        **recalls,
        "spearman": float(spearman) if spearman is not None else float("nan"),
        "n_misranked_pairs": n_misranked,
        "total_pairs": total_pairs,
        "rank_error_frac": n_misranked / total_pairs,
        "MAE": mae_v,
        "RMSE": rmse_v,
        "DeltaE_DFT_mean_misranked": float(np.mean(delta_dft_misranked)) if delta_dft_misranked else 0.0,
        "DeltaE_DFT_max_misranked": float(np.max(delta_dft_misranked)) if delta_dft_misranked else 0.0,
    }


def compute_ordering_summary(dft_ordering: dict, fp_ordering: dict, mode: str,
                                 recall_ks: tuple = (3, 10)) -> dict:
    """
    Macro-averaged ordering metrics over all complete groups for one FP/mode,
    plus energy-error MAE/RMSE pooled across all points of complete groups
    (matching the manuscript's "MAE over all structures in the ordering
    benchmark" -- pooled over points, not averaged over groups, distinct from
    the per-group rank metrics which ARE macro-averaged as specified).

    Reports incomplete groups explicitly rather than silently excluding them
    without a trace.
    """
    per_group = {}
    incomplete_groups = {}
    for comp, dft_group in dft_ordering.items():
        fp_group = fp_ordering[mode].get(comp, {})
        r = compute_ordering_group_metrics(dft_group, fp_group, mode, recall_ks)
        if r is None:
            continue
        if not r["complete"]:
            incomplete_groups[comp] = r["missing_ordered_names"]
            continue
        per_group[comp] = r

    def macro(key):
        vals = [g[key] for g in per_group.values() if key in g and g[key] == g[key]]
        return float(np.mean(vals)) if vals else float("nan")

    keys = ["top1_acc", "spearman", "rank_error_frac",
            "DeltaE_DFT_mean_misranked", "DeltaE_DFT_max_misranked"]
    for k in recall_ks:
        keys.append(f"recall@{k}")

    all_errs = np.concatenate([
        np.array([fp_ordering[mode][comp][n]["energy_per_atom"] for n in dft_ordering[comp]["orderings"]])
        - np.array([dft_ordering[comp]["orderings"][n]["energy_per_atom"] for n in dft_ordering[comp]["orderings"]])
        for comp in per_group
    ]) if per_group else np.array([])
    global_mae, global_rmse = mae_rmse(all_errs) if len(all_errs) else (float("nan"), float("nan"))

    return {
        "n_groups_total": len(dft_ordering),
        "n_groups_complete": len(per_group),
        "n_groups_incomplete": len(incomplete_groups),
        "incomplete_groups": incomplete_groups,
        **{f"avg_{k}": macro(k) for k in keys},
        "global_MAE": global_mae,
        "global_RMSE": global_rmse,
        "N_points": len(all_errs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# M. Data-structure demonstration: the clean data users actually work with,
#    not the legacy raw formats. Mirrors the small force-data demonstration in
#    the Force_error analysis.
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_hull_system(dft_hull: dict, fp_hull: dict, system_name: str, fp_name: str) -> pd.DataFrame:
    """
    One row per candidate in `system_name`: candidate ID, role, endpoint side,
    phase ID, composition, x, DFT energy/atom, full-relaxation and static FP
    energy/atom, and both protocols' status. Structure objects themselves are
    not shown -- use the direct-lookup examples for those.
    """
    rows = []
    for cid, dft_entry in sorted(dft_hull[system_name].items(), key=lambda kv: kv[1]["x"]):
        relax_entry = fp_hull["relax"][system_name].get(cid, {"status": "missing"})
        static_entry = fp_hull["static"][system_name].get(cid, {"status": "missing"})
        rows.append({
            "candidate_id": cid,
            "role": dft_entry["role"],
            "endpoint_side": dft_entry.get("endpoint_side", ""),
            "phase_id": dft_entry["phase_id"],
            "composition": dft_entry["composition"],
            "x": round(dft_entry["x"], 4),
            "DFT E/atom (eV)": round(dft_entry["energy_per_atom"], 5),
            "relax FP E/atom (eV)": round(relax_entry["energy_per_atom"], 5) if relax_entry.get("status") == "success" else None,
            "static FP E/atom (eV)": round(static_entry["energy_per_atom"], 5) if static_entry.get("status") == "success" else None,
            "relax status": relax_entry["status"],
            "static status": static_entry["status"],
        })
    df = pd.DataFrame(rows).set_index("candidate_id")

    print(f"System: {system_name}  ({len(df)} candidates: "
          f"{(df['role']=='interior').sum()} interior + {(df['role']=='endpoint').sum()} endpoint)")
    def _compact(entry, keys):
        if entry is None:
            return "None"
        shown = {k: (f"<structure, {len(entry[k]['sites'])} sites>" if k in entry and isinstance(entry.get(k), dict) and "sites" in entry[k]
                     else entry.get(k))
                 for k in keys if k in entry}
        return shown

    print(f"\nDirect lookup examples (structure fields shown compactly, not dumped):")
    example_cid = df.index[len(df) // 2]
    dft_entry = dft_hull[system_name][example_cid]
    print(f"  dft_hull[{system_name!r}][{example_cid!r}]  ->  "
          f"{_compact(dft_entry, ['role', 'x', 'energy_per_atom', 'relaxed_structure'])}")
    print(f"  fp_hull[{fp_name!r}]['relax'][{system_name!r}][{example_cid!r}]  ->  "
          f"{_compact(fp_hull['relax'][system_name].get(example_cid), ['status', 'energy_per_atom', 'relaxed_structure'])}")
    print(f"  fp_hull[{fp_name!r}]['static'][{system_name!r}][{example_cid!r}]  ->  "
          f"{_compact(fp_hull['static'][system_name].get(example_cid), ['status', 'energy_per_atom'])}")
    return df


def demonstrate_ordering_group(dft_ordering: dict, fp_ordering: dict, group_name: str, fp_name: str) -> pd.DataFrame:
    """
    All 20 configurations of `group_name`: original ordered_name, DFT rank,
    DFT energy/atom, full-relaxation FP energy/atom + rank, static FP
    energy/atom + rank, both protocols' status.
    """
    dft_group = dft_ordering[group_name]["orderings"]
    relax_group = fp_ordering["relax"].get(group_name, {})
    static_group = fp_ordering["static"].get(group_name, {})

    ordered_names = list(dft_group.keys())
    dft_e = np.array([dft_group[n]["energy_per_atom"] for n in ordered_names])
    dft_ranks = pd.Series(dft_e).rank().astype(int).values

    rows = []
    for name, dft_rank in zip(ordered_names, dft_ranks):
        relax_entry = relax_group.get(name, {"status": "missing"})
        static_entry = static_group.get(name, {"status": "missing"})
        rows.append({
            "ordered_name": name,
            "DFT rank": dft_rank,
            "DFT E/atom (eV)": round(dft_group[name]["energy_per_atom"], 5),
            "relax FP E/atom (eV)": round(relax_entry["energy_per_atom"], 5) if relax_entry.get("status") == "success" else None,
            "static FP E/atom (eV)": round(static_entry["energy_per_atom"], 5) if static_entry.get("status") == "success" else None,
            "relax status": relax_entry["status"],
            "static status": static_entry["status"],
        })
    df = pd.DataFrame(rows)
    for col, src in [("relax FP rank", "relax FP E/atom (eV)"), ("static FP rank", "static FP E/atom (eV)")]:
        df[col] = df[src].rank() if df[src].notna().any() else None
    df = df.sort_values("DFT rank").set_index("ordered_name")
    cols = ["DFT rank", "DFT E/atom (eV)",
            "relax FP E/atom (eV)", "relax FP rank", "static FP E/atom (eV)", "static FP rank",
            "relax status", "static status"]
    df = df[cols]

    print(f"Ordering group: {group_name}  (system={dft_ordering[group_name]['system']}, "
          f"phase_id={dft_ordering[group_name]['phase_id']}, {len(df)} configurations)")
    print(f"\nDirect lookup examples:")
    example_name = ordered_names[0]
    print(f"  dft_ordering[{group_name!r}]['orderings'][{example_name!r}]  ->  "
          f"energy_per_atom={dft_group[example_name]['energy_per_atom']:.5f}")
    print(f"  fp_ordering[{fp_name!r}]['relax'][{group_name!r}][{example_name!r}]  ->  "
          f"{relax_group.get(example_name)}")
    print(f"  fp_ordering[{fp_name!r}]['static'][{group_name!r}][{example_name!r}]  ->  "
          f"{static_group.get(example_name)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# N. FPBench data validation summary. Prints actual populations/consistency
#    checks; never forces expected numbers -- prints exact affected
#    identifiers and raises where a hard requirement is violated.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_phase_stability_ordering_results(dft_hull: dict, fp_hull_by_fp: dict,
                            dft_ordering: dict, fp_ordering_by_fp: dict,
                            fps: list[str]) -> dict:
    """
    Prints and returns a validation report:
      - hull: n systems, n unique candidates, n interior, n unique endpoints
      - ordering: n groups, orderings-per-group distribution, n records
      - common candidate IDs / ordering groups / ordered_names across all FPs
        and both protocols
      - missing/failed/non-converged counts per FP and protocol (hull and
        ordering), both as a total and broken down by exact status label
    Raises ValueError on structural violations (e.g. a group without exactly
    20 orderings); reports (does not raise on) missing/incomplete records,
    since those are legitimate real-world outcomes to surface, not bugs.
    """
    report = {}

    # ── Hull population ─────────────────────────────────────────────────────
    # Interior candidate_id only needs to be unique within its own system
    # (dft_hull[system] is a dict, so that is already structurally
    # guaranteed); the same candidate_id string may legitimately repeat
    # across two different systems, so uniqueness is not asserted globally
    # here. n_unique_interior_candidates therefore reports distinct
    # (system, candidate_id) pairs, not distinct candidate_id strings alone.
    n_systems = len(dft_hull)
    interior_pairs = {(s, cid) for s in dft_hull for cid, e in dft_hull[s].items() if e["role"] == "interior"}
    endpoint_ids = {cid for s in dft_hull for cid, e in dft_hull[s].items() if e["role"] == "endpoint"}
    n_interior_appearances = len(interior_pairs)
    report["hull"] = {
        "n_systems": n_systems,
        "n_unique_interior_candidates": n_interior_appearances,
        "n_interior_appearances": n_interior_appearances,
        "n_unique_endpoints": len(endpoint_ids),
        "n_total_unique_candidates": n_interior_appearances + len(endpoint_ids),
    }

    # ── Ordering population ─────────────────────────────────────────────────
    group_sizes = {comp: len(g["orderings"]) for comp, g in dft_ordering.items()}
    bad_groups = {c: n for c, n in group_sizes.items() if n != 20}
    if bad_groups:
        raise ValueError(f"Groups without exactly 20 orderings: {bad_groups}")
    report["ordering"] = {
        "n_groups": len(dft_ordering),
        "orderings_per_group": "20 (all groups, enforced)",
        "n_records": sum(group_sizes.values()),
    }

    # ── Common candidates / groups across all FPs and both protocols ───────
    # Dedup by stable candidate identity: interior candidates are scoped to
    # their own system (system, cid); endpoints are deduped by cid alone
    # since the same endpoint appears in every system it borders.
    def _identity_set(fp_hull, mode):
        out = set()
        for s in fp_hull[mode]:
            for cid, e in fp_hull[mode][s].items():
                if e.get("status") != "success":
                    continue
                identity = cid if dft_hull[s][cid]["role"] == "endpoint" else (s, cid)
                out.add(identity)
        return out

    common_hull_candidates = {}
    for mode in ("relax", "static"):
        sets = [_identity_set(fp_hull_by_fp[fp], mode) for fp in fps]
        common_hull_candidates[mode] = set.intersection(*sets) if sets else set()
    report["hull_common_across_all_fps"] = {
        mode: len(common_hull_candidates[mode]) for mode in common_hull_candidates
    }

    common_ordering_groups = {}
    for mode in ("relax", "static"):
        sets = []
        for fp in fps:
            fo = fp_ordering_by_fp[fp][mode]
            complete = {comp for comp, orderings in fo.items()
                        if all(orderings.get(n, {}).get("status") == "success" for n in dft_ordering[comp]["orderings"])}
            sets.append(complete)
        common_ordering_groups[mode] = set.intersection(*sets) if sets else set()
    report["ordering_common_complete_groups_across_all_fps"] = {
        mode: len(common_ordering_groups[mode]) for mode in common_ordering_groups
    }

    # ── Missing/failed/non-converged per FP and protocol ────────────────────
    # A record's status is one of "success", "missing", "failed",
    # "non_converged" (see _NON_SUCCESS_STATUSES); only "success" is ever
    # scored. hull_missing_by_fp / ordering_missing_by_fp report the total
    # non-success count per FP/protocol (any of the three labels, matching
    # this report's original meaning); hull_status_breakdown_by_fp /
    # ordering_status_breakdown_by_fp break that same total down by the
    # exact status label, so e.g. a genuine calculation failure is never
    # conflated with a record that was simply never attempted.
    def _status_counts(statuses):
        counts = {}
        for st in statuses:
            if st != "success":
                counts[st if st in _NON_SUCCESS_STATUSES else "missing"] = \
                    counts.get(st if st in _NON_SUCCESS_STATUSES else "missing", 0) + 1
        return counts

    hull_missing = {}
    hull_status_breakdown = {}
    for fp in fps:
        fp_hull = fp_hull_by_fp[fp]
        hull_status_breakdown[fp] = {
            mode: _status_counts(e.get("status") for s in fp_hull[mode] for cid, e in fp_hull[mode][s].items())
            for mode in ("relax", "static")
        }
        hull_missing[fp] = {
            mode: sum(hull_status_breakdown[fp][mode].values())
            for mode in ("relax", "static")
        }
    report["hull_missing_by_fp"] = hull_missing
    report["hull_status_breakdown_by_fp"] = hull_status_breakdown

    ordering_missing = {}
    ordering_status_breakdown = {}
    for fp in fps:
        fo = fp_ordering_by_fp[fp]
        ordering_status_breakdown[fp] = {
            mode: _status_counts(e.get("status") for comp in fo[mode] for n, e in fo[mode][comp].items())
            for mode in ("relax", "static")
        }
        ordering_missing[fp] = {
            mode: sum(ordering_status_breakdown[fp][mode].values())
            for mode in ("relax", "static")
        }
    report["ordering_missing_by_fp"] = ordering_missing
    report["ordering_status_breakdown_by_fp"] = ordering_status_breakdown

    # ── Print ────────────────────────────────────────────────────────────────
    print("═" * 78)
    print("FPBENCH DATA VALIDATION SUMMARY")
    print("═" * 78)
    print(f"\nHull:")
    for k, v in report["hull"].items():
        print(f"  {k}: {v}")
    print(f"\nOrdering:")
    for k, v in report["ordering"].items():
        print(f"  {k}: {v}")
    print(f"\nCommon hull candidates across all {len(fps)} FPs (both protocols must have status=success):")
    for mode, n in report["hull_common_across_all_fps"].items():
        print(f"  {mode}: {n} / {report['hull']['n_total_unique_candidates']}")
    print(f"\nCommon COMPLETE ordering groups across all {len(fps)} FPs:")
    for mode, n in report["ordering_common_complete_groups_across_all_fps"].items():
        print(f"  {mode}: {n} / {report['ordering']['n_groups']}")
    print(f"\nMissing hull records by FP/protocol:")
    for fp, d in hull_missing.items():
        print(f"  {fp:20s}  relax={d['relax']:4d}  static={d['static']:4d}")
    print(f"\nMissing ordering records by FP/protocol:")
    for fp, d in ordering_missing.items():
        print(f"  {fp:20s}  relax={d['relax']:4d}  static={d['static']:4d}")
    print(f"\nNon-success status breakdown by FP/protocol (hull):")
    for fp, d in hull_status_breakdown.items():
        for mode in ("relax", "static"):
            if d[mode]:
                print(f"  {fp:20s}  {mode:6s}  " + ", ".join(f"{k}={v}" for k, v in sorted(d[mode].items())))
    print(f"\nNon-success status breakdown by FP/protocol (ordering):")
    for fp, d in ordering_status_breakdown.items():
        for mode in ("relax", "static"):
            if d[mode]:
                print(f"  {fp:20s}  {mode:6s}  " + ", ".join(f"{k}={v}" for k, v in sorted(d[mode].items())))
    print("═" * 78)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# O. One-call manuscript-reproduction loader (wraps FORMAT A/B loading + legacy
#    shape + the standardized data model)
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_convexhull_ordering_data(
    format_a_dirs: dict,
    old_data_dir,
    old_mlips_include: set,
    dft_reference_fp: str,
    ordering_case_alias_compositions: set,
) -> dict:
    """
    LEGACY CONVERSION / PROVENANCE UTILITY -- not the recommended public
    interface. This function reads FPBench's own historical, personal-path
    raw files (FORMAT A/FORMAT B) and is used only by
    scripts/convert_legacy_phase_stability_ordering_data.py, which serializes
    its output into the two canonical standardized files
    (data/phase_stability_ordering_reference.json.gz and
    data/phase_stability_ordering_results_standardized.json.gz). The normal
    analysis notebook path loads those two files directly via
    load_standardized_reference / load_standardized_results and
    build_phase_stability_ordering_results -- it does not call this function.
    Kept here (rather than moved into the conversion script) only so the
    conversion script has a single, already-validated source of truth to
    reconstruct from, without duplicating the FORMAT A/FORMAT B parsing logic.

    Loads every raw FORMAT A / FORMAT B file for the convex-hull and ordering
    benchmarks and builds both a legacy plotting-only shape (kept only for
    provenance cross-checks in the conversion script -- the notebook's own
    demonstration plot instead consumes dft_hull/fp_hull directly via
    _hull_to_legacy_merged, Section P) and the standardized data model
    (dft_hull / fp_hull / dft_ordering / fp_ordering -- everything else is
    computed from these).

    Parameters
    ----------
    format_a_dirs : {fp_name: Path}, e.g. {"mace": Path(...), "mace_matpes_pbe": Path(...)}
        Each directory must contain ALL_convexhull_relaxation.json,
        ALL_convexhull_static.json, ALL_endpoints_relaxation.json,
        ALL_endpoints_static.json, GROUPED20_relax_static.json.
    old_data_dir : Path
        Directory containing all_MLIP_convex_hull_merged.json,
        ALL_MLIP_endpoints_merged.json, ordering_json_files/.
    old_mlips_include : set of FP names to pull from the FORMAT B files.
    dft_reference_fp : which FP's own files to treat as the canonical DFT
        reference (must be a FORMAT A entry -- FORMAT B never stores structures).
    ordering_case_alias_compositions : the confirmed case-alias duplicate
        compositions to deduplicate in the ordering data (see build_dft_ordering).

    Returns
    -------
    dict with keys:
      dft_hull, fp_hull, dft_ordering, fp_ordering        -- the standardized data model
      all_mlips_relax_with_ep, all_mlips_static_with_ep   -- legacy shape (Section 5 plots)
      hull_final_systems                                   -- the 22 (or however many) valid systems
      benchmark_fps                                         -- FP names present in both fp_hull and fp_ordering
    """
    old_hull_json      = old_data_dir / "all_MLIP_convex_hull_merged.json"
    old_endpoints_json = old_data_dir / "ALL_MLIP_endpoints_merged.json"
    old_ordering_dir   = old_data_dir / "ordering_json_files"

    all_mlips_clean_relax, all_mlips_clean_static = {}, {}
    endpoints_relax, endpoints_static = {}, {}
    hull_relax_records_by_fp, hull_static_records_by_fp = {}, {}
    ep_relax_records_by_fp, ep_static_records_by_fp = {}, {}
    ordering_raw_by_fp = {}

    print("[FORMAT A] Loading per-FP flat JSONs ...")
    for mlip_name, data_dir in format_a_dirs.items():
        with open(data_dir / "ALL_convexhull_relaxation.json")  as f: flat_relax    = json.load(f)
        with open(data_dir / "ALL_convexhull_static.json")      as f: flat_static   = json.load(f)
        with open(data_dir / "ALL_endpoints_relaxation.json")   as f: flat_ep_relax = json.load(f)
        with open(data_dir / "ALL_endpoints_static.json")       as f: flat_ep_static= json.load(f)

        _relax, _static = build_clean_from_flat(flat_relax, flat_static)
        all_mlips_clean_relax[mlip_name]  = _relax
        all_mlips_clean_static[mlip_name] = _static
        endpoints_relax[mlip_name]        = build_endpoints_from_flat(flat_ep_relax)
        endpoints_static[mlip_name]       = build_endpoints_from_flat(flat_ep_static)

        hull_relax_records_by_fp[mlip_name]  = _extract_hull_records_format_a(flat_relax)
        hull_static_records_by_fp[mlip_name] = _extract_hull_records_format_a(flat_static)
        ep_relax_records_by_fp[mlip_name]    = _extract_endpoints_format_a(flat_ep_relax)
        ep_static_records_by_fp[mlip_name]   = _extract_endpoints_format_a(flat_ep_static)

        ord_path = data_dir / "GROUPED20_relax_static.json"
        if ord_path.exists():
            with open(ord_path) as f:
                ordering_raw_by_fp[mlip_name] = json.load(f)
        print(f"  {mlip_name}: OK")

    print("[FORMAT B] Loading old nested JSONs ...")
    with open(old_hull_json)      as f: all_merged_raw    = json.load(f)
    with open(old_endpoints_json) as f: all_endpoints_raw = json.load(f)

    old_relax_all  = build_clean_relax(all_merged_raw)
    old_static_all = build_clean_static(all_merged_raw, old_relax_all)
    old_ep_relax   = build_endpoints_relax(all_endpoints_raw)
    old_ep_static  = build_endpoints_static(all_endpoints_raw)

    for mlip_name in old_mlips_include:
        if mlip_name not in old_relax_all:
            print(f"  WARNING: {mlip_name} not found in old hull JSON — skipping")
            continue
        all_mlips_clean_relax[mlip_name]  = old_relax_all[mlip_name]
        all_mlips_clean_static[mlip_name] = old_static_all[mlip_name]
        endpoints_relax[mlip_name]        = old_ep_relax.get(mlip_name, {})
        endpoints_static[mlip_name]       = old_ep_static.get(mlip_name, {})

        hull_relax_records_by_fp[mlip_name]  = _extract_hull_records_format_b(
            all_merged_raw[mlip_name]["MLIP_full_relaxation_initials.json"], want_fp_structure=True)
        hull_static_records_by_fp[mlip_name] = _extract_hull_records_format_b(
            all_merged_raw[mlip_name]["MLIP_static_finals.json"], want_fp_structure=False)
        ep_relax_records_by_fp[mlip_name]  = _extract_endpoints_format_b(
            all_endpoints_raw["MLIP_full_relaxation_initials_endpoints.json"][mlip_name])
        ep_static_records_by_fp[mlip_name] = _extract_endpoints_format_b(
            all_endpoints_raw["MLIP_finals_static_endpoints.json"][mlip_name], structure_key="mp_structure")

        ord_path = old_ordering_dir / f"{mlip_name}_GROUPED20_relax_static.json"
        if ord_path.exists():
            with open(ord_path) as f:
                ordering_raw_by_fp[mlip_name] = json.load(f)
        print(f"  {mlip_name}: OK")

    print("Merging endpoints into legacy hull dicts (Section 5 plots only) ...")
    all_mlips_relax_with_ep  = merge_endpoints_into_hull(all_mlips_clean_relax,  endpoints_relax)
    all_mlips_static_with_ep = merge_endpoints_into_hull(all_mlips_clean_static, endpoints_static)

    print("Building standardized FPBench data model (dft_hull / fp_hull / dft_ordering / fp_ordering) ...")
    dft_hull, hull_final_systems, hull_diag = build_dft_hull(
        hull_relax_records_by_fp[dft_reference_fp], ep_relax_records_by_fp[dft_reference_fp]
    )
    print(f"  dft_hull: {hull_diag}")

    fp_hull = {
        fp: build_fp_hull(
            dft_hull, hull_final_systems,
            hull_relax_records_by_fp[fp], hull_static_records_by_fp[fp],
            ep_relax_records_by_fp[fp], ep_static_records_by_fp[fp],
        )
        for fp in hull_relax_records_by_fp
    }

    dft_ordering = build_dft_ordering(ordering_raw_by_fp[dft_reference_fp], ordering_case_alias_compositions)
    print(f"  dft_ordering: {len(dft_ordering)} groups, "
          f"{sum(len(g['orderings']) for g in dft_ordering.values())} records")

    fp_ordering = {fp: build_fp_ordering(dft_ordering, ordering_raw_by_fp[fp]) for fp in ordering_raw_by_fp}

    benchmark_fps = [fp for fp in all_mlips_clean_relax if fp in fp_hull and fp in fp_ordering]

    return {
        "dft_hull": dft_hull, "fp_hull": fp_hull,
        "dft_ordering": dft_ordering, "fp_ordering": fp_ordering,
        "all_mlips_relax_with_ep": all_mlips_relax_with_ep,
        "all_mlips_static_with_ep": all_mlips_static_with_ep,
        "all_mlips_clean_relax": all_mlips_clean_relax,
        "hull_final_systems": hull_final_systems,
        "benchmark_fps": benchmark_fps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P. Convex-hull metrics demonstration (per-metric highlight figures, modularized)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Renders the same relax/static hull-pair panel as plot_hull_pair, with points
# for one phase (or one composition) emphasized and a checkmark/X badge showing
# whether DFT and the FP agree on that specific metric. Purely a visual
# demonstration of what the Section 6 aggregate numbers mean concretely -- the
# underlying agreement checks (single-phase/composition exclusion, normalized
# energies) are identical to ground_state_agreement_pooled /
# within_phase_hull_min_agreement / global_hull_min_agreement.

import matplotlib.patheffects as _pe
import matplotlib.colors as _mcolors
from matplotlib.lines import Line2D as _Line2D


def _phase_repr_entry(all_merged, mlip, system_name, base_id):
    """Structure of `base_id` closest to an endmember, for labeling."""
    dsys = all_merged[mlip][system_name]
    pts = build_points_for_system(all_merged, mlip, system_name)
    cand = [p for p in pts if extract_base(p["key"]) == base_id]
    if not cand:
        return None
    best = min(cand, key=lambda p: min(p["x"], 1.0 - p["x"]))
    return dsys.get(best["key"])


def _phase_label(all_merged, mlip, system_name, base_id):
    entry = _phase_repr_entry(all_merged, mlip, system_name, base_id)
    if entry is None:
        return base_id
    comp = entry.get("composition")
    comp_str = str(comp.reduced_formula) if comp else base_id
    try:
        struct = entry.get("structure")
        sg = struct.get_space_group_info()[0]
        return f"{comp_str} ({sg})"
    except Exception:
        return comp_str


def _lighten_color(color, amount=0.5):
    """Blend `color` toward white; amount=0 → unchanged, 1 → white."""
    r, g, b = _mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def _valid_points(all_merged, mlip, system_name, include_endpoints=True):
    raw_pts = build_points_for_system(all_merged, mlip, system_name)
    points, _, _ = normalize_to_endpoints(raw_pts)
    valid = [p for p in points if p.get("Erel_dft") is not None and p.get("Erel_mlip") is not None]
    if not include_endpoints:
        valid = [p for p in valid if not p.get("is_endpoint")]
    return valid


def _within_phase_best(all_merged, mlip, system_name, base_id, include_endpoints=True):
    """DFT's and FP's own lowest-Erel structure within one phase. None if <2 candidates
    (matches within_phase_hull_min_agreement's single-composition exclusion)."""
    valid = _valid_points(all_merged, mlip, system_name, include_endpoints)
    phase_pts = [p for p in valid if extract_base(p["key"]) == base_id]
    if len(phase_pts) < 2:
        return None, None, None
    dft_best = min(phase_pts, key=lambda p: p["Erel_dft"])
    mlip_best = min(phase_pts, key=lambda p: p["Erel_mlip"])
    return dft_best, mlip_best, dft_best["key"] == mlip_best["key"]


def _global_hull_min_single(all_merged, mlip, system_name, include_endpoints=True):
    """DFT's and FP's lowest-Erel structure across ALL phases in this one system."""
    valid = _valid_points(all_merged, mlip, system_name, include_endpoints)
    if not valid:
        return None, None, None
    dft_min = min(valid, key=lambda p: p["Erel_dft"])
    mlip_min = min(valid, key=lambda p: p["Erel_mlip"])
    return dft_min, mlip_min, dft_min["key"] == mlip_min["key"]


def _ground_state_match_single(all_merged, mlip, system_name, target_x, x_tol=1e-4, include_endpoints=True):
    """Ground-state check restricted to one composition (tie-line x). None if <2
    competing phases (matches ground_state_agreement_pooled's single-phase exclusion)."""
    valid = _valid_points(all_merged, mlip, system_name, include_endpoints)
    group = [p for p in valid if abs(p["x"] - target_x) <= x_tol]
    if len(group) < 2:
        return None
    dft_best = min(group, key=lambda p: p["Erel_dft"])
    mlip_best = min(group, key=lambda p: p["Erel_mlip"])
    return dft_best["key"] == mlip_best["key"]


def list_shared_compositions(all_merged, mlip, system_name, x_tol=1e-4):
    """Composition x-values where more than one phase (mp-id) is present --
    candidates for target_x in the ground-state demonstration figure."""
    raw_pts = build_points_for_system(all_merged, mlip, system_name)
    points, _, _ = normalize_to_endpoints(raw_pts)
    xs = sorted({round(p["x"], 4) for p in points})
    groups = []
    for x in xs:
        grp = [p for p in points if abs(p["x"] - x) <= x_tol]
        bases = sorted({extract_base(p["key"]) for p in grp})
        if len(bases) > 1:
            groups.append((x, bases))
    return groups


def _badge(ax, x, y, ok, fontsize=14):
    if ok is None:
        symbol, color = "–", "gray"
    elif ok:
        symbol, color = "✓", "green"
    else:
        symbol, color = "✗", "crimson"
    ax.text(x, y, symbol, transform=ax.transAxes, fontsize=fontsize, color=color,
             fontweight="bold", ha="left", va="top",
             path_effects=[_pe.withStroke(linewidth=2.5, foreground="white")])


def _phase_hull_lines(ax, all_merged, mlip, system_name, target_phase, mlip_hull_color, dft_hull_color,
                       lighten_amount, lw):
    """Lower hull built from ONLY this phase's own points, in a lighter tint of
    the global hull colors, so it reads as 'this phase's local hull'."""
    raw_pts = build_points_for_system(all_merged, mlip, system_name)
    points, _, _ = normalize_to_endpoints(raw_pts)
    phase_pts = [p for p in points if extract_base(p["key"]) == target_phase]
    mlip_xy = [(p["x"], p["Erel_mlip"]) for p in phase_pts if p.get("Erel_mlip") is not None]
    dft_xy = [(p["x"], p["Erel_dft"]) for p in phase_pts if p.get("Erel_dft") is not None]
    mlip_color = _lighten_color(mlip_hull_color, lighten_amount)
    dft_color = _lighten_color(dft_hull_color, lighten_amount)
    handles = []
    if len(mlip_xy) >= 2:
        Hm = lower_hull(mlip_xy)
        ax.plot([t[0] for t in Hm], [t[1] for t in Hm], lw=lw, color=mlip_color, zorder=3)
        handles.append(_Line2D([0], [0], color=mlip_color, lw=lw, label="FP hull (this phase)"))
    if len(dft_xy) >= 2:
        Hd = lower_hull(dft_xy)
        ax.plot([t[0] for t in Hd], [t[1] for t in Hd], lw=lw, color=dft_color, zorder=3)
        handles.append(_Line2D([0], [0], color=dft_color, lw=lw, label="DFT hull (this phase)"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)


def _match_collections_to_points(ax, points, dodge, atol=1e-6):
    """Pair each scatter artist on `ax` back to the point dict that produced it,
    by matching plotted (x, y) coordinates."""
    matches = []
    for coll in ax.collections:
        offs = coll.get_offsets()
        if offs.shape[0] != 1:
            continue
        ox, oy = offs[0]
        found = None
        for p in points:
            em = p.get("Erel_mlip")
            if em is not None and abs(ox - (p["x"] - dodge)) < atol and abs(oy - em) < atol:
                found = p
                break
            ed = p.get("Erel_dft")
            if ed is not None and abs(ox - (p["x"] + dodge)) < atol and abs(oy - ed) < atol:
                found = p
                break
        if found is not None:
            matches.append((coll, found))
    return matches


def _apply_emphasis(ax, all_merged, mlip, system_name, dodge, is_emph, fade_alpha, emph_lw, emph_s):
    """Fade every point that fails `is_emph`; leave matching points at full opacity."""
    raw_pts = build_points_for_system(all_merged, mlip, system_name)
    points, _, _ = normalize_to_endpoints(raw_pts)
    matches = _match_collections_to_points(ax, points, dodge)
    for coll, p in matches:
        if is_emph(p):
            coll.set_alpha(1.0)
            coll.set_linewidths([emph_lw])
            coll.set_sizes([emph_s])
        else:
            coll.set_alpha(fade_alpha)
    return points


_HULL_DEMO_DEFAULTS = dict(
    mlip_marker="x", dft_marker="+", mlip_hollow=True, dft_hollow=True,
    mlip_alpha=0.9, dft_alpha=1.0, mlip_lw=1.0, dft_lw=1.0, dodge=0.003,
    label_by="composition+spacegroup", hull_lw=1.0, hull_alpha=1.0,
    show_zero_line=True, zero_line_alpha=0.25, suptitle_y=0.985, suptitle_x=0.54,
    marker_label="separate", mlip_hull_color="darkorange", dft_hull_color="black",
    phase_colors=None, figsize=(6.4, 3.3), legend_fontsize=9,
    relax_title="Full FP", static_title="Static FP",
    fade_alpha=0.15, emph_lw=1.4, emph_s=50, lighten_amount=0.55, badge_fontsize=16,
)


def _render_hull_pair(all_merged_relax, all_merged_static, mlip, system_name, model_names, opts):
    fig, axes = plot_hull_pair(
        all_merged_relax, all_merged_static, mlip, system_name,
        plot_fn=plot_normalized_with_hulls,
        figsize=opts["figsize"],
        mlip_marker=opts["mlip_marker"], dft_marker=opts["dft_marker"],
        mlip_hollow=opts["mlip_hollow"], dft_hollow=opts["dft_hollow"],
        mlip_alpha=opts["mlip_alpha"], dft_alpha=opts["dft_alpha"],
        mlip_lw=opts["mlip_lw"], dft_lw=opts["dft_lw"],
        mlip_display=model_names.get(mlip, mlip),
        display_system_name=None,
        dodge=opts["dodge"], label_by=opts["label_by"],
        hull_lw=opts["hull_lw"], hull_alpha=opts["hull_alpha"],
        show_zero_line=opts["show_zero_line"], zero_line_alpha=opts["zero_line_alpha"],
        suptitle_y=opts["suptitle_y"], suptitle_x=opts["suptitle_x"],
        marker_label=opts["marker_label"],
        mlip_hull_color=opts["mlip_hull_color"], dft_hull_color=opts["dft_hull_color"],
        phase_colors=opts["phase_colors"], legend_fontsize=opts["legend_fontsize"],
        relax_title=opts["relax_title"], static_title=opts["static_title"],
        legend_phase_ncol=3, legend_marker_ncol=2, legend_hull_ncol=2,
    )
    if fig.legends:
        fig.legends[0].set_bbox_to_anchor((0.5, -0.31), transform=fig.transFigure)
        plt.subplots_adjust(top=0.85, bottom=0.10)
    # Static-FP panel shares the y-axis range with Full-FP (sharey=True) but
    # matplotlib hides its tick labels by default; show the actual numbers
    # there instead of repeating the "$E_f$ (eV/atom)" text on both panels.
    axes[1].tick_params(labelleft=True)
    axes[1].set_ylabel("")
    return fig, axes


def plot_within_phase_demo(all_merged_relax, all_merged_static, mlip, system_name, target_phase,
                            model_names, include_endpoints=True, **kwargs):
    """Demonstration figure for within_phase_hull_min_agreement: emphasizes one
    phase's points and badges whether DFT/FP agree on its lowest-energy composition."""
    opts = {**_HULL_DEMO_DEFAULTS, **kwargs}
    fig, axes = _render_hull_pair(all_merged_relax, all_merged_static, mlip, system_name, model_names, opts)
    is_emph = lambda p: extract_base(p["key"]) == target_phase
    for ax, all_merged in zip(axes, (all_merged_relax, all_merged_static)):
        _apply_emphasis(ax, all_merged, mlip, system_name, opts["dodge"], is_emph,
                         opts["fade_alpha"], opts["emph_lw"], opts["emph_s"])
        _phase_hull_lines(ax, all_merged, mlip, system_name, target_phase,
                           opts["mlip_hull_color"], opts["dft_hull_color"], opts["lighten_amount"], opts["hull_lw"])
        _, _, match = _within_phase_best(all_merged, mlip, system_name, target_phase, include_endpoints)
        _badge(ax, 0.03, 0.97, match, fontsize=opts["badge_fontsize"])
    fig.suptitle(f"{format_system_name(system_name)} — within-phase best-structure",
                 fontsize=13, y=opts["suptitle_y"])
    plt.show()
    return fig, axes


def plot_hull_min_demo(all_merged_relax, all_merged_static, mlip, system_name,
                        model_names, include_endpoints=True, **kwargs):
    """Demonstration figure for global_hull_min_agreement: badges whether DFT/FP
    agree on the system-wide minimum-energy structure."""
    opts = {**_HULL_DEMO_DEFAULTS, **kwargs}
    fig, axes = _render_hull_pair(all_merged_relax, all_merged_static, mlip, system_name, model_names, opts)
    for ax, all_merged in zip(axes, (all_merged_relax, all_merged_static)):
        _, _, match = _global_hull_min_single(all_merged, mlip, system_name, include_endpoints)
        _badge(ax, 0.03, 0.97, match, fontsize=opts["badge_fontsize"])
    fig.suptitle(f"{format_system_name(system_name)} — Global convex hull-minimum agreement",
                 fontsize=13, y=opts["suptitle_y"])
    plt.show()
    return fig, axes


def plot_ground_state_demo(all_merged_relax, all_merged_static, mlip, system_name, target_x,
                            model_names, x_tol=1e-4, include_endpoints=True, **kwargs):
    """Demonstration figure for ground_state_agreement_pooled: emphasizes the
    competing phases at one composition and badges whether DFT/FP agree."""
    opts = {**_HULL_DEMO_DEFAULTS, **kwargs}
    fig, axes = _render_hull_pair(all_merged_relax, all_merged_static, mlip, system_name, model_names, opts)
    is_emph = lambda p: abs(p["x"] - target_x) <= x_tol
    for ax, all_merged in zip(axes, (all_merged_relax, all_merged_static)):
        _apply_emphasis(ax, all_merged, mlip, system_name, opts["dodge"], is_emph,
                         opts["fade_alpha"], opts["emph_lw"], opts["emph_s"])
        ok = _ground_state_match_single(all_merged, mlip, system_name, target_x, x_tol, include_endpoints)
        _badge(ax, 0.03, 0.97, ok, fontsize=opts["badge_fontsize"])
    fig.suptitle(f"{format_system_name(system_name)} — ground-state agreement at x={target_x:.3f}",
                 fontsize=13, y=opts["suptitle_y"])
    plt.show()
    return fig, axes


def _hull_to_legacy_merged(dft_hull: dict, fp_hull_by_fp: dict, mode: str) -> dict:
    """
    Build a legacy-shaped {fp: {system: {candidate_id: {...}}}} dict directly
    from the standardized dft_hull/fp_hull, entirely in memory, for feeding
    plot_normalized_with_hulls / build_points_for_system (which still consume
    that shape, per "do not rewrite the historical plotting-only pipeline
    unless necessary"). This never reads a raw file and never calls
    load_all_convexhull_ordering_data -- it is the adapter that lets the
    demonstration plot consume dft_hull/fp_hull directly.

    Per-entry fields match what build_points_for_system reads: "structure"
    (a pymatgen Structure, not a dict -- structure_frac_dict needs the
    object), "MLIP_energy(/atom)", "DFT_energy(/atom)". "sid" carries
    phase_id (used only for legend colour grouping via extract_base, which
    for every existing candidate_id already returns exactly phase_id).
    build_points_for_system recomputes x geometrically from "structure"
    itself, so dft_hull's own stored "x" is not needed here.
    """
    out: dict = {}
    for fp, fp_hull in fp_hull_by_fp.items():
        out[fp] = {}
        for system, candidates in dft_hull.items():
            out[fp][system] = {}
            for cid, dft_entry in candidates.items():
                struct = Structure.from_dict(dft_entry["relaxed_structure"])
                rec = {
                    "structure": struct,
                    "n_atoms": dft_entry["n_atoms"],
                    "composition": struct.composition,
                    "sid": dft_entry["phase_id"],
                    "DFT_energy": dft_entry["energy_total"],
                    "DFT_energy/atom": dft_entry["energy_per_atom"],
                }
                fp_entry = fp_hull.get(mode, {}).get(system, {}).get(cid)
                if fp_entry is not None and fp_entry.get("status") == "success":
                    rec["MLIP_energy"] = fp_entry["energy_total"]
                    rec["MLIP_energy/atom"] = fp_entry["energy_per_atom"]
                out[fp][system][cid] = rec
    return out


def plot_hull_pair_demo(
    dft_hull, fp_hull_by_fp, mlip, system_name, model_names,
    mlip_marker="x", dft_marker="+", mlip_hollow=True, dft_hollow=True,
    mlip_alpha=0.9, dft_alpha=1.0, mlip_lw=1.0, dft_lw=1.0, dodge=0.003,
    label_by="composition+spacegroup", hull_lw=1.0, hull_alpha=1.0,
    show_zero_line=True, zero_line_alpha=0.25,
    suptitle_y=0.985, suptitle_x=0.54,
    marker_label="separate", mlip_hull_color="darkorange", dft_hull_color="black",
    phase_colors=None,
    mlip_label_fontweight="normal", mlip_label_alpha=1.0,
    dft_label_fontweight="normal", dft_label_alpha=1.0,
    axis_label_fontweight="normal",
    relax_title="Full FP relaxation", static_title="Static FP",
    show_endpoint_labels=True, endpoint_labels=None,
    endpoint_label_fontsize=9, endpoint_label_fontweight="normal",
    endpoint_label_y=-0.10,
    endpoint_label_left_dx=-0.03, endpoint_label_left_dy=0.0,
    endpoint_label_right_dx=0.03, endpoint_label_right_dy=0.0,
    figsize=(6.4, 3.3), legend_fontsize=9,
    subplot_left=None, subplot_right=None, panel_wspace=0.2,
    legend_phase_ncol=3, legend_marker_ncol=2, legend_hull_ncol=2,
):
    """Plain relax/static hull-pair panel, no metric badges -- the main
    single-system comparison figure. Consumes dft_hull/fp_hull_by_fp (the
    standardized data structures) directly -- internally builds the
    relax/static legacy-shaped dicts plot_normalized_with_hulls still expects
    via _hull_to_legacy_merged, entirely in memory. Per-phase coloured
    scatter; endmember labels (e.g. "GeSe2" / "SiSe2") sit under x=0/x=1
    instead of a redundant suptitle; Static FP panel shows its own y-axis
    numbers (shares the Full FP panel's range via sharey, but matplotlib
    hides tick labels on shared axes by default)."""
    all_merged_relax = _hull_to_legacy_merged(dft_hull, fp_hull_by_fp, "relax")
    all_merged_static = _hull_to_legacy_merged(dft_hull, fp_hull_by_fp, "static")
    fig, axes = plot_hull_pair(
        all_merged_relax, all_merged_static, mlip, system_name,
        plot_fn=plot_normalized_with_hulls,
        figsize=figsize,
        mlip_marker=mlip_marker, dft_marker=dft_marker,
        mlip_hollow=mlip_hollow, dft_hollow=dft_hollow,
        mlip_alpha=mlip_alpha, dft_alpha=dft_alpha,
        mlip_lw=mlip_lw, dft_lw=dft_lw,
        mlip_display=model_names.get(mlip, mlip),
        display_system_name=None,   # endpoint labels already show the two end members
        dodge=dodge, label_by=label_by,
        hull_lw=hull_lw, hull_alpha=hull_alpha,
        show_zero_line=show_zero_line, zero_line_alpha=zero_line_alpha,
        suptitle_y=suptitle_y, suptitle_x=suptitle_x,
        marker_label=marker_label,
        mlip_hull_color=mlip_hull_color, dft_hull_color=dft_hull_color,
        phase_colors=phase_colors,
        mlip_label_fontweight=mlip_label_fontweight, mlip_label_alpha=mlip_label_alpha,
        dft_label_fontweight=dft_label_fontweight, dft_label_alpha=dft_label_alpha,
        axis_label_fontweight=axis_label_fontweight,
        legend_fontsize=legend_fontsize,
        subplot_left=subplot_left, subplot_right=subplot_right,
        relax_title=relax_title, static_title=static_title,
        show_endpoint_labels=show_endpoint_labels, endpoint_labels=endpoint_labels,
        endpoint_label_fontsize=endpoint_label_fontsize, endpoint_label_fontweight=endpoint_label_fontweight,
        endpoint_label_y=endpoint_label_y,
        endpoint_label_left_dx=endpoint_label_left_dx, endpoint_label_left_dy=endpoint_label_left_dy,
        endpoint_label_right_dx=endpoint_label_right_dx, endpoint_label_right_dy=endpoint_label_right_dy,
        legend_phase_ncol=legend_phase_ncol, legend_marker_ncol=legend_marker_ncol, legend_hull_ncol=legend_hull_ncol,
        panel_wspace=panel_wspace,
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=True)
    if fig.legends:
        fig.legends[0].set_bbox_to_anchor((0.5, -0.31), transform=fig.transFigure)
        plt.subplots_adjust(top=0.85, bottom=0.10)
    plt.show()
    return fig, axes


def demonstrate_hull_metrics(all_merged_relax, all_merged_static, mlip, system_name,
                              target_phase, target_x, model_names, include_endpoints=True, **kwargs):
    """One call, three figures: within-phase, global hull-minimum, and
    ground-state agreement, all for the same (mlip, system_name)."""
    plot_within_phase_demo(all_merged_relax, all_merged_static, mlip, system_name, target_phase,
                            model_names, include_endpoints, **kwargs)
    plot_hull_min_demo(all_merged_relax, all_merged_static, mlip, system_name,
                        model_names, include_endpoints, **kwargs)
    plot_ground_state_demo(all_merged_relax, all_merged_static, mlip, system_name, target_x,
                            model_names, x_tol=1e-4, include_endpoints=include_endpoints, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Q. Ordering metrics demonstration (single-group energy-level schematic, modularized)
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_positions(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def compute_group_ranking_metrics(E_dft, E_fp, k=3):
    """Same is_misranked_pair definition as compute_ordering_group_metrics,
    plus the individual misranked pairs (needed to annotate the plot)."""
    from itertools import combinations
    from scipy.stats import spearmanr

    E_dft, E_fp = np.asarray(E_dft, dtype=float), np.asarray(E_fp, dtype=float)
    n = len(E_dft)
    dft_rank, fp_rank = np.argsort(E_dft), np.argsort(E_fp)

    misranked_pairs = [
        (i, j, abs(E_dft[i] - E_dft[j]))
        for i, j in combinations(range(n), 2)
        if is_misranked_pair(E_dft[i], E_dft[j], E_fp[i], E_fp[j])
    ]
    return {
        "top1_acc": int(dft_rank[0] == fp_rank[0]),
        f"recall@{k}": len(set(dft_rank[:k]) & set(fp_rank[:k])) / k,
        "spearman": spearmanr(E_dft, E_fp).correlation,
        "rank_error_frac": len(misranked_pairs) / (n * (n - 1) / 2),
        "misranked_pairs": misranked_pairs,
    }


def find_demo_ordering_group(dft_ordering, fp_ordering_by_fp, fps, mode="relax"):
    """First ordering group (composition) with a complete FP result for every
    FP in `fps`. Returns the composition key, or None if none qualify."""
    for comp in sorted(dft_ordering):
        if all(
            all(fp_ordering_by_fp[fp][mode][comp][n]["status"] == "success" for n in dft_ordering[comp]["orderings"])
            for fp in fps
        ):
            return comp
    return None


def plot_ranking_schematic(
    E_dft, E_fp, k=3, group_label="", fp_label="", save_name=None,
    figsize=(6.2, 8.0),
    ranking_errors_x=-0.62, ranking_errors_y_frac=0.15,
    misranked_label_dx=0.0, misranked_label_dy=0.0,
    lowest_energy_label_frac=0.12,
):
    """
    DFT-vs-FP energy-level schematic for one ordering group: dashed red
    connectors mark structures whose relative rank flips between DFT and the
    FP; the annotated pair is the most severely misranked (largest ΔE_DFT).

    Energies are plotted as-is, in meV/atom (converted from the eV/atom
    inputs) -- NOT shifted to a per-side minimum. DFT and FP per-atom
    energies for the *same* structure are already on a directly comparable
    absolute scale (unlike the tie-line formation energies used elsewhere in
    this notebook), so no artificial baseline is needed, and the ΔE_DFT arrow
    below genuinely spans the two actual DFT bar heights of the misranked
    pair it annotates.

    Position knobs (defaults can make "Ranking errors" and "ΔE_DFT of
    misranked pair" cross for some groups -- nudge these per group):
      ranking_errors_x, ranking_errors_y_frac : "Ranking errors" text
        position (x in the same -0.7..1.45 data coordinates as the DFT/FP
        columns; y_frac is a fraction of the plotted y-range, measured up
        from the bottom of the range).
      misranked_label_dx, misranked_label_dy : nudge for the "ΔE_DFT of
        misranked pair" text away from its default position next to the
        arrow (dx in data-x units, dy as a fraction of the y-range).
      lowest_energy_label_frac : how far below the lowest plotted energy the
        "Lowest-energy configurations" text and its arrows start, as a
        fraction of the y-range -- smaller = higher up (closer to the bars).
    """
    E_dft_meV = np.asarray(E_dft, dtype=float) * 1000.0
    E_fp_meV  = np.asarray(E_fp, dtype=float) * 1000.0
    metrics = compute_group_ranking_metrics(E_dft_meV, E_fp_meV, k=k)

    dft_ranks, fp_ranks = _rank_positions(E_dft_meV), _rank_positions(E_fp_meV)
    x_dft, x_fp, bar_width = 0.0, 1.0, 0.24

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(E_dft_meV)):
        rank_changed = dft_ranks[i] != fp_ranks[i]
        connector_color = "red" if rank_changed else "0.55"
        connector_alpha = 0.75 if rank_changed else 0.35
        connector_lw = 1.8 if rank_changed else 1.2
        ax.hlines(E_dft_meV[i], x_dft - bar_width / 2, x_dft + bar_width / 2, color="#1f77b4", lw=3)
        ax.hlines(E_fp_meV[i], x_fp - bar_width / 2, x_fp + bar_width / 2, color="#ff7f0e", lw=3)
        ax.plot([x_dft + bar_width / 2, x_fp - bar_width / 2], [E_dft_meV[i], E_fp_meV[i]], "--",
                color=connector_color, alpha=connector_alpha, lw=connector_lw)

    dft_lec, fp_lec = np.argmin(E_dft_meV), np.argmin(E_fp_meV)
    ax.scatter([x_dft], [E_dft_meV[dft_lec]], s=90, color="#1f77b4", edgecolor="black", zorder=5)
    ax.scatter([x_fp], [E_fp_meV[fp_lec]], s=90, color="#ff7f0e", edgecolor="black", zorder=5)

    y_min = min(E_dft_meV.min(), E_fp_meV.min())
    y_max = max(E_dft_meV.max(), E_fp_meV.max())
    y_range = (y_max - y_min) or 1.0

    if metrics["misranked_pairs"]:
        i, j, dE = max(metrics["misranked_pairs"], key=lambda x: x[2])
        y1, y2 = E_dft_meV[i], E_dft_meV[j]   # exactly the two blue (DFT) bar heights for this pair
        ax.annotate("", xy=(x_dft - 0.32, y1), xytext=(x_dft - 0.32, y2),
                    arrowprops=dict(arrowstyle="<->", color="red", lw=2))
        ax.text(x_dft - 0.43 + misranked_label_dx, 0.5 * (y1 + y2) + misranked_label_dy * y_range,
                r"$\Delta E_{\mathrm{DFT}}$" + "\nof misranked pair",
                color="red", ha="right", va="center", fontsize=12)

    changed_indices = np.where(dft_ranks != fp_ranks)[0]
    if len(changed_indices) > 0:
        idx = changed_indices[0]
        mid_y = 0.5 * (E_dft_meV[idx] + E_fp_meV[idx])
        ax.annotate("Ranking\nerrors", xy=(0.5, mid_y),
                    xytext=(ranking_errors_x, y_min + ranking_errors_y_frac * y_range),
                    arrowprops=dict(arrowstyle="->", lw=2, linestyle="--", color="black"),
                    ha="center", va="center", fontsize=18, color="black")

    text_y = y_min - lowest_energy_label_frac * y_range
    ax.annotate("", xy=(x_dft, E_dft_meV[dft_lec]), xytext=(0.5, text_y + 0.04 * y_range),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(x_fp, E_fp_meV[fp_lec]), xytext=(0.5, text_y + 0.04 * y_range),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(0.5, text_y, "Lowest-energy\nconfigurations", ha="center", va="top", fontsize=17)

    ax.set_xlim(-0.7, 1.45)
    ax.set_ylim(text_y - 0.10 * y_range, y_max + 0.18 * y_range)
    ax.set_xticks([x_dft, x_fp])
    ax.set_xticklabels([r"$E^{\mathrm{DFT}}$", r"$E^{\mathrm{FP}}$"], fontsize=17)
    ax.set_ylabel("Energy (meV/atom)", fontsize=18)
    ax.set_title(
        f"{group_label}\n{fp_label}\n"
        f"Top-1 = {metrics['top1_acc']}   Recall@{k} = {metrics[f'recall@{k}']:.2f}\n"
        f"Spearman ρ = {metrics['spearman']:.2f}   Rate of ranking errors = {metrics['rank_error_frac']:.2f}",
        fontsize=12, pad=15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_name is not None:
        fig.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_name}.svg", bbox_inches="tight")
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
    plt.show()
    return fig, ax


def demonstrate_ordering_ranking(dft_ordering, fp_ordering, group_name, fp_name, model_names,
                                  mode="relax", k=3, save_name=None, **kwargs):
    """One call: pulls DFT/FP energies for `group_name` and renders the
    ranking schematic. `format_system_name` is used for the group label.
    Pass plot_ranking_schematic's position knobs (ranking_errors_x,
    ranking_errors_y_frac, misranked_label_dx, misranked_label_dy,
    lowest_energy_label_frac) as extra keyword arguments if the default
    layout crosses for a given group."""
    ordered_names = sorted(dft_ordering[group_name]["orderings"].keys())
    E_dft = np.array([dft_ordering[group_name]["orderings"][n]["energy_per_atom"] for n in ordered_names])
    E_fp = np.array([fp_ordering[mode][group_name][n]["energy_per_atom"] for n in ordered_names])
    group_label = " · ".join(format_system_name(p) for p in
                              [dft_ordering[group_name]["system"], dft_ordering[group_name]["phase_id"]])
    return plot_ranking_schematic(
        E_dft, E_fp, k=k, group_label=group_label,
        fp_label=model_names.get(fp_name, fp_name), save_name=save_name, **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# R. Results-table builders (hull, ordering, RMSD) — one call each
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Average energy error" is reported as MAE alone, matching the manuscript
# (Tables 4 and 6 show a single MAE value, not an MAE/RMSE pair -- that
# convention is specific to the Force_error component, not this one). RMSE is
# still computed internally (available via the *_row helper functions below,
# e.g. for anyone who wants it) but is not part of the displayed table.

def _hull_metrics_row(dft_hull, fp_hull_one_fp, mode):
    """Numeric metrics for one FP, one mode. MAE/RMSE both returned (table
    builder below displays MAE only, per the manuscript's own convention)."""
    errors = []
    for system, candidates in dft_hull.items():
        for cid, dft_entry in candidates.items():
            fp_entry = fp_hull_one_fp[mode][system].get(cid)
            if fp_entry is None or fp_entry.get("status") != "success":
                continue
            errors.append(fp_entry["energy_per_atom"] - dft_entry["energy_per_atom"])
    arr = np.array(errors)
    mae, rmse = mae_rmse(arr)

    gs = ground_state_agreement_pooled(dft_hull, fp_hull_one_fp, mode)
    wp = within_phase_hull_min_agreement(dft_hull, fp_hull_one_fp, mode)
    gh = global_hull_min_agreement(dft_hull, fp_hull_one_fp, mode)
    return {"MAE": mae, "RMSE": rmse, "GP_Frac": gs["fraction"],
            "BaseBest_Frac": wp["fraction"], "Hull_Min_Match": gh["fraction"],
            "_gs": gs, "_wp": wp, "_gh": gh}


def build_combined_hull_table(dft_hull, fp_hull, fps, model_names,
                               decimal_places=2, mev_fmt="{:.0f}"):
    """
    Combined Full-FP-relaxation + Static hull results table, formatted for
    display. Returns (combined_table, rows_by_mode) -- rows_by_mode carries
    the raw numerator/denominator detail (_gs/_wp/_gh) for anyone who wants
    to print or inspect it separately.
    """
    fmt_pct = f"{{:.{decimal_places}f}}"

    def _format(rows_by_fp):
        out = pd.DataFrame(index=[model_names.get(fp, fp) for fp in fps])
        out["Average energy error (meV/atom)"] = [
            mev_fmt.format(rows_by_fp[fp]["MAE"] * 1000) for fp in fps
        ]
        out["Ground- state agreement (%)"] = [
            fmt_pct.format(rows_by_fp[fp]["GP_Frac"] * 100) for fp in fps
        ]
        out["Within-phase hull-minimum agreement (%)"] = [
            fmt_pct.format(rows_by_fp[fp]["BaseBest_Frac"] * 100) for fp in fps
        ]
        out["Hull-minimum agreement (%)"] = [
            fmt_pct.format(rows_by_fp[fp]["Hull_Min_Match"] * 100) for fp in fps
        ]
        return out

    rows_by_mode = {
        mode: {fp: _hull_metrics_row(dft_hull, fp_hull[fp], mode) for fp in fps}
        for mode in ("relax", "static")
    }
    combined = pd.concat(
        [_format(rows_by_mode["relax"]), _format(rows_by_mode["static"])], axis=1,
        keys=["Full FP relaxation", "Static FP evaluations on the DFT-relaxed structures"],
    )
    combined.index.name = "FP"
    return combined, rows_by_mode


def build_combined_ordering_table(dft_ordering, fp_ordering, fps, model_names,
                                   decimal_places=1, mev_fmt="{:.0f}"):
    """
    Combined Full-FP-relaxation + Static ordering results table, formatted
    for display. Returns (combined_table, summaries) -- summaries carries the
    full per-FP compute_ordering_summary output (group completeness, etc.)
    """
    summaries = {
        fp: {mode: compute_ordering_summary(dft_ordering, fp_ordering[fp], mode) for mode in ("relax", "static")}
        for fp in fps
    }

    def _format(mode):
        out = pd.DataFrame(index=[model_names.get(fp, fp) for fp in fps])
        rows = {fp: summaries[fp][mode] for fp in fps}
        out["Average energy error (meV/atom) ↓"] = [mev_fmt.format(rows[fp]["global_MAE"] * 1000) for fp in fps]
        out["Top-1 accuracy (%) ↑"] = [round(rows[fp]["avg_top1_acc"] * 100, decimal_places) for fp in fps]
        out["Recall@3 (%) ↑"] = [round(rows[fp]["avg_recall@3"] * 100, decimal_places) for fp in fps]
        out["Recall@10 (%) ↑"] = [round(rows[fp]["avg_recall@10"] * 100, decimal_places) for fp in fps]
        out["Spearman ρ ↑"] = [round(rows[fp]["avg_spearman"], decimal_places) for fp in fps]
        out["Rate of ranking errors (%) ↓"] = [round(rows[fp]["avg_rank_error_frac"] * 100, decimal_places) for fp in fps]
        out["Mean/max ΔE_DFT (meV/atom) ↓"] = [
            f"{mev_fmt.format(rows[fp]['avg_DeltaE_DFT_mean_misranked']*1000)} / "
            f"{mev_fmt.format(rows[fp]['avg_DeltaE_DFT_max_misranked']*1000)}"
            for fp in fps
        ]
        return out

    combined = pd.concat(
        [_format("relax"), _format("static")],
        keys=["Full FP relaxation", "Static FP evaluations on the DFT-relaxed structures"],
    )
    combined.index.names = ["", "FP"]
    return combined, summaries


def build_rmsd_table(dft_hull, fp_hull, fps, model_names, mode="relax", decimal_places=2):
    """Combined RMSD results table (relax only -- matches the manuscript,
    which reports RMSD for full FP relaxation, not static). Returns
    (table, rmsd_dfs, summaries)."""
    fmt = f"{{:.{decimal_places}f}}"
    rmsd_dfs = {fp: compute_structure_rmsd(dft_hull, fp_hull[fp], mode) for fp in fps}
    summaries = {fp: summarize_structure_rmsd(rmsd_dfs[fp]) for fp in fps}

    table = pd.DataFrame(index=[model_names.get(fp, fp) for fp in fps])
    table["Map success (%)"] = [fmt.format(summaries[fp]["map_success_pct"]) for fp in fps]
    table["Mean/max RMSD (Å)"] = [
        f"{fmt.format(summaries[fp]['avg_rmsd_ang'])} / {fmt.format(summaries[fp]['max_rmsd_ang'])}" for fp in fps
    ]
    table["RMSD < 0.05 Å (%)"] = [fmt.format(summaries[fp]["rmsd_lt_0.05_pct"]) for fp in fps]
    table["RMSD < 0.10 Å (%)"] = [fmt.format(summaries[fp]["rmsd_lt_0.10_pct"]) for fp in fps]
    table["RMSD < 0.20 Å (%)"] = [fmt.format(summaries[fp]["rmsd_lt_0.20_pct"]) for fp in fps]
    table.index.name = "FP"
    return table, rmsd_dfs, summaries


# ═══════════════════════════════════════════════════════════════════════════════
# S. Public builder for external datasets. Normalizes user-supplied DFT
#    reference data + FP results (plain Python structures, not FPBench's
#    historical FORMAT A/FORMAT B file layout) into the same standardized
#    FPBench data structures Sections I-R already consume. Reuses those
#    structures and every downstream metric/table function unchanged; does
#    not duplicate any metric logic. load_all_convexhull_ordering_data
#    (Section O) is untouched and remains the manuscript-reproduction loader
#    for the historical files.
# ═══════════════════════════════════════════════════════════════════════════════

# Recognized non-"success" status labels a caller may supply for an FP result.
# Only "success" records are ever scored; "missing"/"failed"/"non_converged"
# are all excluded from scoring identically, but the distinct label supplied
# by the caller is preserved through to fp_hull/fp_ordering (and reported
# separately by validate_phase_stability_ordering_results) rather than being
# collapsed to a single generic "missing". Any other/absent status string is
# normalized to "missing" (an unrecognized status is treated as "no result",
# not silently scored).
_NON_SUCCESS_STATUSES = ("missing", "failed", "non_converged")


def _n_atoms_and_epa(rec: dict) -> tuple[int, float]:
    """(n_atoms, energy_per_atom) for one user-supplied record. n_atoms comes
    from the record's own "n_atoms" field if given, else is derived from its
    "relaxed_structure" -- one of the two is required."""
    n_atoms = rec.get("n_atoms")
    if n_atoms is None:
        struct_dict = rec.get("relaxed_structure")
        if struct_dict is None:
            raise ValueError("record has neither 'n_atoms' nor a 'relaxed_structure' to derive it from")
        n_atoms = len(Structure.from_dict(struct_dict))
    return n_atoms, rec["energy_total"] / n_atoms


def build_phase_stability_ordering_results(reference_data: dict, fp_results: dict) -> dict:
    """
    Public, format-agnostic builder for the standardized FPBench data
    structures (dft_hull, fp_hull, dft_ordering, fp_ordering), for datasets
    that do not follow FPBench's historical FORMAT A/FORMAT B file layout.
    See load_all_convexhull_ordering_data for that (unchanged) manuscript-
    reproduction loader.

    Returns the exact same object shapes Sections I-R already work with, so
    the result can be passed straight to build_combined_hull_table,
    build_combined_ordering_table, build_rmsd_table, and
    validate_phase_stability_ordering_results with no further conversion --
    this function does not reimplement or duplicate any of those functions'
    metric logic, it only assembles their required input shape from a
    simpler, natural-identifier input.

    Parameters
    ----------
    reference_data : {
        "hull": {                              # optional -- omit if no hull benchmark
            system_name: {
                candidate_id: {
                    "role": "interior" | "endpoint",         # required
                    "phase_id": str,                          # required
                    "composition": str,                       # required, free-form label
                    "energy_total": float,                    # required, eV
                    "relaxed_structure": dict,                # required, pymatgen Structure.as_dict()
                    "initial_structure": dict | None,         # optional
                    "n_atoms": int,                           # optional; derived from relaxed_structure
                    "x": float | None,                        # optional, provenance only -- see note below
                    "endpoint_side": "left" | "right",        # required iff role == "endpoint"
                }, ...
            }, ...
        },
        "ordering": {                           # optional -- omit if no ordering benchmark
            group_key: {
                "system": str | None,                         # optional, provenance only
                "phase_id": str,                               # required
                "composition": str,                            # required
                "orderings": {                                 # exactly 20 entries required
                    ordered_name: {
                        "energy_total": float,                 # required, eV
                        "n_atoms": int,                        # optional; derived from relaxed_structure
                        "relaxed_structure": dict | None,      # optional, provenance only
                        "initial_structure": dict | None,      # optional
                    }, ...
                },
            }, ...
        },
    }
    fp_results : {
        fp_name: {
            "hull": {                           # keyed to match reference_data["hull"]
                "relax":  {system_name: {candidate_id: {"status": "success"|"missing"|"failed"|"non_converged",
                                                          "energy_total": float,
                                                          "relaxed_structure": dict}}},
                "static": {system_name: {candidate_id: {"status": "success"|"missing"|"failed"|"non_converged",
                                                          "energy_total": float}}},
            },
            "ordering": {                       # keyed to match reference_data["ordering"]
                "relax":  {group_key: {ordered_name: {"status": ..., "energy_total": float}}},
                "static": {group_key: {ordered_name: {"status": ..., "energy_total": float}}},
            },
        }, ...
    }

    A candidate/ordered_name absent from fp_results, or explicitly marked
    "missing"/"failed"/"non_converged", is recorded with that same status
    (absent or unrecognized -> "missing") -- never backfilled from the DFT
    energy or structure. Only status="success" records are ever scored;
    "missing"/"failed"/"non_converged" are excluded identically but the
    distinct label is preserved for validate_phase_stability_ordering_results
    to report separately. Per-atom energies are always computed here from
    energy_total and atom count (never taken as a caller-supplied field), so
    DFT and every FP are normalized the same way by construction.

    Identifier requirements (validated up front, not silently worked around)
    --------------------------------------------------------------------------
    - candidate_id only needs to be stable and unique within its documented
      scope (unique among a system's own candidates for interior candidates;
      unique among all endpoints for endpoints) -- it does not need to encode
      phase_id in any way. Phase grouping (within_phase_hull_min_agreement,
      Section J) reads the explicit phase_id field directly.
    - system_name should be "{formula_A}_{formula_B}" (two pymatgen-parsable
      chemical formulas) if within_phase_hull_min_agreement /
      global_hull_min_agreement are to be meaningful for it: those two
      metrics recompute each candidate's tie-line fraction geometrically from
      its own relaxed structure and the two endmember formulas parsed out of
      system_name -- the "x" field above is stored for reference only and is
      NOT what those two metrics use. Average energy error and ground-state
      agreement do not depend on this convention and stay meaningful for any
      system_name. A system that doesn't decompose along a two-endmember
      line is simply excluded from within-phase/global hull-minimum
      agreement (reported as 0/0, i.e. NaN) rather than raising.
    - every system_name / group_key referenced under fp_results must already
      exist in reference_data -- an unrecognized key is treated as a likely
      naming mismatch and raises, rather than being silently treated as "no
      result for this FP" (a candidate/ordered_name missing within a known
      system/group is the legitimate way to express that).

    Returns
    -------
    {"dft_hull": ..., "fp_hull": ..., "dft_ordering": ..., "fp_ordering": ...}
    """
    hull_ref = reference_data.get("hull", {})
    ordering_ref = reference_data.get("ordering", {})

    # ── dft_hull ─────────────────────────────────────────────────────────────
    dft_hull: dict = {}
    for system, candidates in hull_ref.items():
        dft_hull[system] = {}
        for cid, rec in candidates.items():
            for required in ("role", "phase_id", "composition", "energy_total", "relaxed_structure"):
                if required not in rec:
                    raise ValueError(f"reference_data['hull'][{system!r}][{cid!r}] is missing required field {required!r}")
            role = rec["role"]
            if role not in ("interior", "endpoint"):
                raise ValueError(f"hull[{system!r}][{cid!r}]['role'] must be 'interior' or 'endpoint', got {role!r}")
            phase_id = rec["phase_id"]
            if role == "endpoint" and rec.get("endpoint_side") not in ("left", "right"):
                raise ValueError(f"hull[{system!r}][{cid!r}]: endpoint role requires endpoint_side 'left' or 'right'")

            n_atoms, epa = _n_atoms_and_epa(rec)
            entry = {
                "candidate_id": cid,
                "source_keys": {"origin": "user_provided"},
                "role": role,
                "phase_id": phase_id,
                "composition": rec["composition"],
                "x": rec.get("x"),
                "n_atoms": n_atoms,
                "energy_total": rec["energy_total"],
                "energy_per_atom": epa,
                "initial_structure": rec.get("initial_structure"),
                "relaxed_structure": rec["relaxed_structure"],
            }
            if role == "endpoint":
                entry["endpoint_side"] = rec["endpoint_side"]
            dft_hull[system][cid] = entry

    # ── fp_hull ──────────────────────────────────────────────────────────────
    fp_hull: dict = {}
    for fp, payload in fp_results.items():
        fp_hull_in = payload.get("hull", {})
        fp_hull[fp] = {"relax": {}, "static": {}}
        for mode in ("relax", "static"):
            mode_data = fp_hull_in.get(mode, {})
            unknown_systems = set(mode_data) - set(dft_hull)
            if unknown_systems:
                raise ValueError(
                    f"fp_results[{fp!r}]['hull'][{mode!r}] references system(s) not present "
                    f"in reference_data['hull']: {sorted(unknown_systems)}"
                )
            for system, dft_candidates in dft_hull.items():
                fp_hull[fp][mode][system] = {}
                system_results = mode_data.get(system, {})
                for cid, dft_entry in dft_candidates.items():
                    rec = system_results.get(cid)
                    status_in = rec.get("status") if rec is not None else None
                    if rec is None or status_in != "success" or rec.get("energy_total") is None:
                        out_status = status_in if status_in in _NON_SUCCESS_STATUSES else "missing"
                        fp_hull[fp][mode][system][cid] = {"status": out_status}
                        continue
                    if mode == "relax" and not rec.get("relaxed_structure"):
                        raise ValueError(
                            f"fp_results[{fp!r}]['hull']['relax'][{system!r}][{cid!r}] has an "
                            "energy_total but no relaxed_structure -- a successful full-relaxation "
                            "result must carry its own relaxed structure (never backfilled from the "
                            "DFT structure). Mark status='missing' if this FP result doesn't exist."
                        )
                    entry = {
                        "status": "success",
                        "energy_total": rec["energy_total"],
                        "energy_per_atom": rec["energy_total"] / dft_entry["n_atoms"],
                    }
                    if mode == "relax":
                        entry["relaxed_structure"] = rec["relaxed_structure"]
                    fp_hull[fp][mode][system][cid] = entry

    # ── dft_ordering ─────────────────────────────────────────────────────────
    dft_ordering: dict = {}
    for group_key, group in ordering_ref.items():
        for required in ("phase_id", "composition", "orderings"):
            if required not in group:
                raise ValueError(f"reference_data['ordering'][{group_key!r}] is missing required field {required!r}")
        orderings_in = group["orderings"]
        if len(orderings_in) != 20:
            raise ValueError(f"ordering[{group_key!r}] must have exactly 20 orderings, got {len(orderings_in)}")
        orderings = {}
        for name, rec in orderings_in.items():
            if "energy_total" not in rec:
                raise ValueError(f"ordering[{group_key!r}]['orderings'][{name!r}] is missing 'energy_total'")
            n_atoms, epa = _n_atoms_and_epa(rec)
            orderings[name] = {
                "n_atoms": n_atoms,
                "energy_total": rec["energy_total"],
                "energy_per_atom": epa,
                "initial_structure": rec.get("initial_structure"),
                "relaxed_structure": rec.get("relaxed_structure"),
            }
        dft_ordering[group_key] = {
            "system": group.get("system"),
            "phase_id": group["phase_id"],
            "composition": group["composition"],
            "source_group_keys": [group_key],
            "orderings": orderings,
        }

    # ── fp_ordering ──────────────────────────────────────────────────────────
    fp_ordering: dict = {}
    for fp, payload in fp_results.items():
        fp_ord_in = payload.get("ordering", {})
        fp_ordering[fp] = {"relax": {}, "static": {}}
        for mode in ("relax", "static"):
            mode_data = fp_ord_in.get(mode, {})
            unknown_groups = set(mode_data) - set(dft_ordering)
            if unknown_groups:
                raise ValueError(
                    f"fp_results[{fp!r}]['ordering'][{mode!r}] references group(s) not present "
                    f"in reference_data['ordering']: {sorted(unknown_groups)}"
                )
            for group_key, dft_group in dft_ordering.items():
                fp_ordering[fp][mode][group_key] = {}
                group_results = mode_data.get(group_key, {})
                for name, dft_o in dft_group["orderings"].items():
                    rec = group_results.get(name)
                    status_in = rec.get("status") if rec is not None else None
                    if rec is None or status_in != "success" or rec.get("energy_total") is None:
                        out_status = status_in if status_in in _NON_SUCCESS_STATUSES else "missing"
                        fp_ordering[fp][mode][group_key][name] = {"status": out_status}
                        continue
                    fp_ordering[fp][mode][group_key][name] = {
                        "status": "success",
                        "energy_total": rec["energy_total"],
                        "energy_per_atom": rec["energy_total"] / dft_o["n_atoms"],
                    }

    return {
        "dft_hull": dft_hull,
        "fp_hull": fp_hull,
        "dft_ordering": dft_ordering,
        "fp_ordering": fp_ordering,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# T. Standardized files: the two canonical manuscript-reproduction data
#    products (data/phase_stability_ordering_reference.json.gz and
#    data/phase_stability_ordering_results_standardized.json.gz), a loader for
#    each, and merge_phase_stability_ordering_fp_results for combining
#    standardized single-FP result fragments into the merged "models" mapping
#    build_phase_stability_ordering_results accepts as fp_results. Both
#    standardized files are produced by
#    scripts/convert_legacy_phase_stability_ordering_data.py, never edited by
#    hand.
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_STABILITY_ORDERING_SCHEMA_VERSION = "1.0"
PHASE_STABILITY_ORDERING_DATASET_NAME = "PCM-phase-stability-ordering"

# Legacy (raw-data-era) FP key -> stable canonical model key used in the
# standardized files and in this notebook's own analysis order from here on.
PHASE_STABILITY_ORDERING_MODEL_KEY_MAP = {
    "mace": "mace",
    "CHGNet": "chgnet",
    "M3GNet": "m3gnet_mp",
    "UMA": "uma",
    "M3GNet_MatPES": "m3gnet_matpes_pbe",
    "TensorNet_MatPES": "tensornet_pbe",
    "mace_matpes_pbe": "mace_matpes_pbe",
}

# Canonical analysis order (stable model keys), independent of the legacy
# dict-iteration order any raw file happened to produce.
PHASE_STABILITY_ORDERING_MODEL_ORDER = [
    "mace",
    "chgnet",
    "m3gnet_mp",
    "uma",
    "m3gnet_matpes_pbe",
    "tensornet_pbe",
    "mace_matpes_pbe",
]

# Human-readable manuscript display names, keyed by the stable model key
# (deliberately separate from the model key itself -- the key is a stable
# identifier, this is only for table/figure display).
PHASE_STABILITY_ORDERING_MODEL_NAMES = {
    "mace": "MACE",
    "chgnet": "CHGNet",
    "m3gnet_mp": "M3GNet",
    "uma": "UMA",
    "m3gnet_matpes_pbe": "M3GNet-MatPES",
    "tensornet_pbe": "TensorNet-MatPES",
    "mace_matpes_pbe": "MACE-MatPES",
}


def sha256_of_file(path) -> str:
    """SHA-256 hex digest of a file's raw bytes, read in chunks (safe for the
    large standardized files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_standardized_json_gz(path, payload: dict) -> None:
    """Serialize `payload` as deterministic JSON (sorted keys, fixed
    separators, no NaN/Infinity) and gzip it with a zeroed mtime/filename so
    repeated runs on identical content produce a byte-identical .json.gz --
    used for both canonical standardized files and any merged fragment
    output."""
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(data)


def load_standardized_reference(path) -> dict:
    """
    Load the canonical DFT reference file
    (data/phase_stability_ordering_reference.json.gz). Returns the full
    payload dict: schema_version, component, dataset_name, units,
    reference_metadata, reference_data. Pass payload["reference_data"] as the
    reference_data argument to build_phase_stability_ordering_results.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_standardized_results(path) -> dict:
    """
    Load the canonical merged all-model FP-results file
    (data/phase_stability_ordering_results_standardized.json.gz). Returns the
    full payload dict: schema_version, component, dataset_name, units,
    reference (filename/sha256 of the reference file it was built against),
    generation_metadata, models. Pass payload["models"] as the fp_results
    argument to build_phase_stability_ordering_results.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _load_fragment_payload(fragment):
    """A fragment is either an in-memory dict, or a path (str/Path) to a
    .json/.json.gz file containing one."""
    if isinstance(fragment, dict):
        return fragment
    p = Path(fragment)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_phase_stability_ordering_fp_results(
    fragments,
    reference_data: dict,
    schema_version: str = PHASE_STABILITY_ORDERING_SCHEMA_VERSION,
    dataset_name: str = PHASE_STABILITY_ORDERING_DATASET_NAME,
    reference_sha256: str | None = None,
    existing_models: dict | None = None,
    out_path=None,
) -> dict:
    """
    Combine one or more standardized single-FP result fragments into the
    merged {"models": {...}} structure build_phase_stability_ordering_results
    accepts as fp_results. Intended for the (future) generator workflow,
    where each FP's own run produces one standardized fragment that gets
    added to the common all-model results file -- not for everyday analysis,
    which loads the already-merged file via load_standardized_results.

    Parameters
    ----------
    fragments : dict[model_key, fragment] | list[fragment] | list[(model_key, fragment)]
        Each fragment is either an in-memory dict or a path (str/Path) to a
        .json/.json.gz file containing one, with the shape:
          {"schema_version": ..., "dataset_name": ...,
           "reference": {"sha256": ...}, "model_key": ...,
           "metadata": {...},
           "hull": {"relax": {system: {candidate_id: {...}}},
                     "static": {system: {candidate_id: {...}}}},
           "ordering": {"relax": {group: {ordered_name: {...}}},
                        "static": {group: {ordered_name: {...}}}}}
        model_key is taken from the fragment's own "model_key" field when
        present, else from the dict/tuple key supplied alongside it -- if
        both are given they must agree.
    reference_data : the same reference_data dict passed to
        build_phase_stability_ordering_results (i.e. a loaded standardized
        reference file's payload["reference_data"]) -- used only to validate
        that every candidate_id / ordered_name referenced by a fragment
        already exists in the shared DFT reference universe.
    schema_version, dataset_name, reference_sha256 : the exact values every
        fragment (and this merge) must match. A fragment with a different
        schema_version, dataset_name, or reference sha256 raises rather than
        being silently merged against a reference it was not computed
        against.
    existing_models : an already-loaded standardized results payload (i.e.
        load_standardized_results(...)'s return value) or a bare
        {"models": {...}} dict to extend. None starts from an empty
        "models" mapping. A model_key already present in either
        `existing_models` or an earlier fragment in this same call raises --
        this function never silently overwrites an existing model.
    out_path : optional path to also write the merged payload as a
        standardized results .json.gz file (via write_standardized_json_gz).

    Returns
    -------
    {"schema_version": ..., "component": "phase_stability_ordering",
     "dataset_name": ..., "units": {"energy": "eV"},
     "reference": {"sha256": ...}, "generation_metadata": {...},
     "models": {model_key: {"metadata": ..., "hull": ..., "ordering": ...}, ...}}
    -- pass result["models"] as fp_results to build_phase_stability_ordering_results.
    """
    if isinstance(fragments, dict):
        items = list(fragments.items())
    else:
        items = []
        for frag in fragments:
            if isinstance(frag, tuple):
                items.append(frag)
            else:
                loaded = _load_fragment_payload(frag)
                mk = loaded.get("model_key")
                if mk is None:
                    raise ValueError(
                        "fragment has no 'model_key' field and was not supplied as "
                        "(model_key, fragment) -- cannot determine its stable model key"
                    )
                items.append((mk, loaded))

    if existing_models and "models" in existing_models:
        models = dict(existing_models["models"])
    else:
        models = dict(existing_models or {})

    hull_ids = {system: set(cands) for system, cands in reference_data.get("hull", {}).items()}
    ordering_ids = {group: set(g["orderings"]) for group, g in reference_data.get("ordering", {}).items()}

    seen_in_this_call = set()
    for key_from_container, frag in items:
        frag = _load_fragment_payload(frag)
        frag_model_key = frag.get("model_key")
        model_key = frag_model_key if frag_model_key is not None else key_from_container
        if frag_model_key is not None and key_from_container is not None and frag_model_key != key_from_container:
            raise ValueError(
                f"fragment's own model_key {frag_model_key!r} does not match the "
                f"supplied key {key_from_container!r}"
            )
        if model_key in seen_in_this_call:
            raise ValueError(f"duplicate model_key within this merge call: {model_key!r}")
        seen_in_this_call.add(model_key)
        if model_key in models:
            raise ValueError(
                f"model_key {model_key!r} is already present -- never silently "
                "overwritten; remove it from existing_models first if a genuine "
                "replacement is intended"
            )

        if frag.get("schema_version") != schema_version:
            raise ValueError(
                f"{model_key!r}: fragment schema_version {frag.get('schema_version')!r} "
                f"!= expected {schema_version!r}"
            )
        if frag.get("dataset_name") != dataset_name:
            raise ValueError(
                f"{model_key!r}: fragment dataset_name {frag.get('dataset_name')!r} "
                f"!= expected {dataset_name!r}"
            )
        frag_sha = (frag.get("reference") or {}).get("sha256")
        if reference_sha256 is not None and frag_sha != reference_sha256:
            raise ValueError(
                f"{model_key!r}: fragment reference sha256 {frag_sha!r} != expected "
                f"{reference_sha256!r} -- this fragment was not computed against the "
                "same DFT reference file"
            )

        hull = frag.get("hull") or {"relax": {}, "static": {}}
        ordering = frag.get("ordering") or {"relax": {}, "static": {}}

        for mode in ("relax", "static"):
            for system, cands in hull.get(mode, {}).items():
                if system not in hull_ids:
                    raise ValueError(f"{model_key!r}: hull[{mode!r}] references unknown system {system!r}")
                unknown = set(cands) - hull_ids[system]
                if unknown:
                    raise ValueError(
                        f"{model_key!r}: hull[{mode!r}][{system!r}] references candidate_id(s) "
                        f"not present in the reference: {sorted(unknown)}"
                    )
            for group, names in ordering.get(mode, {}).items():
                if group not in ordering_ids:
                    raise ValueError(f"{model_key!r}: ordering[{mode!r}] references unknown group {group!r}")
                unknown = set(names) - ordering_ids[group]
                if unknown:
                    raise ValueError(
                        f"{model_key!r}: ordering[{mode!r}][{group!r}] references ordered_name(s) "
                        f"not present in the reference: {sorted(unknown)}"
                    )

        models[model_key] = {
            "metadata": frag.get("metadata", {}),
            "hull": hull,
            "ordering": ordering,
        }

    merged = {
        "schema_version": schema_version,
        "component": "phase_stability_ordering",
        "dataset_name": dataset_name,
        "units": {"energy": "eV"},
        "reference": {"sha256": reference_sha256},
        "generation_metadata": {"merged_via": "merge_phase_stability_ordering_fp_results"},
        "models": models,
    }

    if out_path is not None:
        write_standardized_json_gz(out_path, merged)

    return merged
