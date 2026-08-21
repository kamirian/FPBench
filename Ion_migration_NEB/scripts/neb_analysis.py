"""Non-plotting analysis helpers for the Ion Migration by NEB FPBench notebook.

Loads, validates, transforms, and calculates analysis results from the
canonical DFT reference and FP results JSON files (see input_data/README.md
and results/README.md for the full schema documentation). Contains no
plotting code (see neb_plots.py) and no legacy-source reading (see
scripts/reconstruct_neb_datasets.py).

Every function accepts its scientific inputs explicitly and returns a
documented result. No function reads a notebook global.

Protocol identifiers (used throughout, matching the canonical JSON schema
verbatim):
    PROTOCOL_FULL_FP_NEB           "full_fp_neb"
    PROTOCOL_FP_STATIC_ON_DFT_NEB  "fp_static_on_dft_neb"
    PROTOCOL_DFT_STATIC_ON_FP_NEB  "dft_static_on_fp_neb"

Units: energies in eV, forces in eV/angstrom, RMSD/distances in angstrom,
force-angle errors in degrees, all as in the canonical JSON schema.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from ase.calculators.singlepoint import SinglePointCalculator

PROTOCOL_FULL_FP_NEB = "full_fp_neb"
PROTOCOL_FP_STATIC_ON_DFT_NEB = "fp_static_on_dft_neb"
PROTOCOL_DFT_STATIC_ON_FP_NEB = "dft_static_on_fp_neb"

MIGRATING_ATOM_INDEX = 0

_ADAPTOR = AseAtomsAdaptor()


# ─────────────────────────────────────────────────────────────────────────
# 1. Load / validate / coverage
# ─────────────────────────────────────────────────────────────────────────

def _normalize_reference_data(reference_data):
    """Accept either DFT-reference schema and return an object every
    function in this module can read via the flat pathways[key]["dft_neb_images"]
    list this module's internals were originally written against:

      - the canonical nested schema
        (pathways[key]["dft_neb_reference"]["images"]); or
      - the legacy flat schema (pathways[key]["dft_neb_images"]) directly.

    Never writes a second serialized file and never duplicates structure/
    energy/force content: when only the nested form is present, the added
    flat list is built by sorting the *same* image dicts already present
    under "dft_neb_reference"."images"] by image_index (a new Python list
    referencing the same dict objects, not a deep copy). Image order,
    endpoint roles, units, and identifiers are read, never altered.

    Returns a new top-level dict (the caller's reference_data is never
    mutated in place); unaffected pathway dicts are passed through by
    reference, not copied.

    Raises ValueError if a pathway supplies both representations and they
    disagree on image count, energy, or forces for any image -- this is
    treated as a real data inconsistency, never silently resolved.
    """
    pathways = reference_data.get("pathways")
    if not isinstance(pathways, dict):
        return reference_data

    normalized_pathways = {}
    for pkey, pdata in pathways.items():
        nested_images = pdata.get("dft_neb_reference", {}).get("images")
        flat_images = pdata.get("dft_neb_images")

        if nested_images is not None and flat_images is not None:
            nested_sorted = sorted(nested_images, key=lambda im: im["image_index"])
            if len(nested_sorted) != len(flat_images):
                raise ValueError(
                    f"{pkey}: dft_neb_reference.images ({len(nested_sorted)} images) and "
                    f"dft_neb_images ({len(flat_images)} images) disagree on image count"
                )
            for i, (n_im, f_im) in enumerate(zip(nested_sorted, flat_images)):
                if n_im.get("energy_total_eV") != f_im.get("energy_total_eV"):
                    raise ValueError(
                        f"{pkey} image {i}: dft_neb_reference.images and dft_neb_images "
                        f"disagree on energy_total_eV"
                    )
                if n_im.get("forces_eV_per_angstrom") != f_im.get("forces_eV_per_angstrom"):
                    raise ValueError(
                        f"{pkey} image {i}: dft_neb_reference.images and dft_neb_images "
                        f"disagree on forces_eV_per_angstrom"
                    )
            normalized_pathways[pkey] = pdata

        elif nested_images is not None:
            new_pdata = dict(pdata)
            new_pdata["dft_neb_images"] = sorted(nested_images, key=lambda im: im["image_index"])
            normalized_pathways[pkey] = new_pdata

        else:
            # Legacy flat-only schema, or neither key present: pass through
            # unchanged. If dft_neb_images is genuinely absent and later
            # code needs it, that code raises its own natural error rather
            # than this function fabricating one preemptively.
            normalized_pathways[pkey] = pdata

    normalized = dict(reference_data)
    normalized["pathways"] = normalized_pathways
    return normalized


def _load_json_maybe_gz(path):
    """json.load a plain .json file, or a gzip-compressed .json.gz file
    (detected by file extension, not content sniffing)."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def load_neb_datasets(dft_reference_file, fp_results_file):
    """Load the two canonical JSON files (input_data/ion_migration_neb_reference.json
    and results/ion_migration_neb_results.json, or any file following the same
    schema). Accepts plain .json or gzip-compressed .json.gz for either file
    (detected by file extension). Returns (dft_reference_data, fp_results_data).
    dft_reference_data is normalized (see _normalize_reference_data) so either
    the canonical nested schema or the legacy flat schema is accepted.
    fp_results_data is returned exactly as loaded: every branch, including the
    "pathways"/"images" branches the rest of this module reads and the sibling
    "unsuccessful_pathways"/"unsuccessful_image_attempts" branches it does not,
    is preserved verbatim -- nothing is filtered, renamed, or dropped here."""
    dft_reference_data = _load_json_maybe_gz(dft_reference_file)
    fp_results_data = _load_json_maybe_gz(fp_results_file)
    dft_reference_data = _normalize_reference_data(dft_reference_data)
    return dft_reference_data, fp_results_data


def validate_neb_datasets(dft_reference_data, fp_results_data, expected_pathway_count, fp_order):
    """Schema/identifier/image/status checks. Returns a dict with a 'summary_df'
    DataFrame (one row per check) and 'ok' (bool, True iff every check passed).
    Never raises on a failed check; every failure is a row in the report."""
    checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        dft_reference_data = _normalize_reference_data(dft_reference_data)
        record("dft_neb_reference/dft_neb_images normalization", True)
    except ValueError as e:
        record("dft_neb_reference/dft_neb_images normalization", False, str(e))

    # 3.1 schema/version
    record("dft_reference has schema_version",
           "schema_version" in dft_reference_data,
           dft_reference_data.get("schema_version"))
    record("fp_results has schema_version",
           "schema_version" in fp_results_data,
           fp_results_data.get("schema_version"))

    # 3.2 pathway/material identifiers
    common_keys = set(dft_reference_data.get("common_pathway_keys", []))
    record(f"common_pathway_keys count == {expected_pathway_count}",
           len(common_keys) == expected_pathway_count, f"found {len(common_keys)}")
    record("dft_reference pathways == common_pathway_keys",
           set(dft_reference_data["pathways"].keys()) == common_keys,
           f"{len(dft_reference_data['pathways'])} pathways present")
    for pkey, pdata in dft_reference_data["pathways"].items():
        ident = pdata.get("identifiers", {})
        if "icsd_id" not in ident or "source_path_id" not in ident:
            record(f"pathway {pkey} has icsd_id/source_path_id", False, str(ident))
            break
    else:
        record("every DFT-reference pathway has icsd_id/source_path_id", True)

    # 3.3 protocol / FP coverage
    models = fp_results_data.get("models", {})
    record("all FPs in fp_order present in models",
           set(fp_order) <= set(models.keys()),
           f"missing: {set(fp_order) - set(models.keys())}")
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB, PROTOCOL_DFT_STATIC_ON_FP_NEB):
            if protocol not in model:
                record(f"{fp_key}.{protocol} present", False)

    # 3.4 image identifiers/ordering (DFT reference)
    bad_ordering = []
    for pkey, pdata in dft_reference_data["pathways"].items():
        imgs = pdata["dft_neb_images"]
        indices = [im["image_index"] for im in imgs]
        if indices != list(range(len(imgs))):
            bad_ordering.append(pkey)
    record("DFT-reference image_index sequences are 0..N-1 in order",
           len(bad_ordering) == 0, f"{len(bad_ordering)} bad pathways: {bad_ordering[:5]}")

    # 3.5 structure/energy/force presence (spot check every pathway, full_fp_neb)
    missing_fields = []
    for fp_key in fp_order:
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            for idx_str, img in pdata["final_fp_neb_images"].items():
                for field in ("structure", "fp_energy_total_eV", "fp_forces_eV_per_angstrom"):
                    if field not in img:
                        missing_fields.append((fp_key, pkey, idx_str, field))
    record("full_fp_neb images have structure/energy/forces", len(missing_fields) == 0,
           f"{len(missing_fields)} missing fields")

    # 3.6 status/convergence checks
    missing_status = []
    for fp_key in fp_order:
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            if "neb_status" not in pdata or "neb_converged" not in pdata["neb_status"]:
                missing_status.append((fp_key, pkey))
    record("full_fp_neb pathways have neb_status.neb_converged", len(missing_status) == 0,
           f"{len(missing_status)} missing")

    summary_df = pd.DataFrame(checks)
    ok = bool(summary_df["ok"].all()) if len(summary_df) else False
    return type("ValidationReport", (), {"summary_df": summary_df, "ok": ok, "checks": checks})()


def collect_unsuccessful_records(fp_results_data, fp_order):
    """{fp_key: {protocol: {record_key: record}}} for every unsuccessful or
    incomplete attempt recorded outside the "pathways"/"images" branches the
    rest of this module reads: full_fp_neb.unsuccessful_pathways and
    fp_static_on_dft_neb.unsuccessful_pathways (each keyed by pathway_key,
    record field "calculation_status"), plus
    dft_static_on_fp_neb.unsuccessful_image_attempts (keyed by
    "{fp_key}|{pathway_key}|{image_index}", record field "status"). Missing
    branches (e.g. an older results file without them) default to {}.

    Purely additive status/coverage reporting: no image-map-building or
    metric function in this module (build_dft_neb_image_map,
    build_full_fp_neb_image_map, build_fp_static_on_dft_neb_image_map,
    build_full_fp_neb_status_map, and everything downstream of them) ever
    reads these branches, so nothing returned here can enter a barrier,
    profile, RMSD, or force-error calculation or change any existing
    denominator."""
    models = fp_results_data.get("models", {})
    out = {}
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        out[fp_key] = {
            PROTOCOL_FULL_FP_NEB: dict(model.get(PROTOCOL_FULL_FP_NEB, {}).get("unsuccessful_pathways", {})),
            PROTOCOL_FP_STATIC_ON_DFT_NEB: dict(model.get(PROTOCOL_FP_STATIC_ON_DFT_NEB, {}).get("unsuccessful_pathways", {})),
            PROTOCOL_DFT_STATIC_ON_FP_NEB: dict(model.get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("unsuccessful_image_attempts", {})),
        }
    return out


def build_protocol_coverage_table(dft_reference_data, fp_results_data, fp_order, fp_display_names=None):
    """Compact per-FP, per-protocol coverage table built from the canonical
    data (not hardcoded). Columns: fp_key, fp_display_name, protocol,
    present_pathway_count, pathway_fp_combination_count, total_image_count,
    image_count_distribution, non_converged_count (full_fp_neb only, else
    'not applicable'), source_population_counts (dft_static_on_fp_neb only,
    else None), unsuccessful_count, unsuccessful_status_counts.

    unsuccessful_count / unsuccessful_status_counts report the sibling
    "unsuccessful_pathways" branch (full_fp_neb, fp_static_on_dft_neb) or
    "unsuccessful_image_attempts" branch (dft_static_on_fp_neb) -- status
    reporting only, read via collect_unsuccessful_records; these records are
    never read by present_pathway_count/total_image_count/non_converged_count
    above or by any scientific metric function in this module."""
    fp_display_names = fp_display_names or {}
    models = fp_results_data.get("models", {})
    unsuccessful_by_fp_protocol = collect_unsuccessful_records(fp_results_data, fp_order)
    rows = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB, PROTOCOL_DFT_STATIC_ON_FP_NEB):
            pathways = model.get(protocol, {}).get("pathways", {})
            n_pathways = len(pathways)
            image_key = "final_fp_neb_images" if protocol == PROTOCOL_FULL_FP_NEB else "images"
            total_images = 0
            dist = {}
            for pdata in pathways.values():
                n = len(pdata[image_key])
                total_images += n
                dist[n] = dist.get(n, 0) + 1

            if protocol == PROTOCOL_FULL_FP_NEB:
                n_not_conv = sum(
                    1 for p in pathways.values()
                    if not p.get("neb_status", {}).get("neb_converged", True)
                )
            else:
                n_not_conv = "not applicable"

            if protocol == PROTOCOL_DFT_STATIC_ON_FP_NEB:
                src_pop_counts = {}
                for p in pathways.values():
                    pop = p.get("source_population", "unknown")
                    src_pop_counts[pop] = src_pop_counts.get(pop, 0) + 1
            else:
                src_pop_counts = None

            unsuccessful = unsuccessful_by_fp_protocol.get(fp_key, {}).get(protocol, {})
            status_field = "status" if protocol == PROTOCOL_DFT_STATIC_ON_FP_NEB else "calculation_status"
            unsuccessful_status_counts = {}
            for urec in unsuccessful.values():
                s = urec.get(status_field, "unknown")
                unsuccessful_status_counts[s] = unsuccessful_status_counts.get(s, 0) + 1

            rows.append({
                "fp_key": fp_key,
                "fp_display_name": fp_display_names.get(fp_key, fp_key),
                "protocol": protocol,
                "present_pathway_count": n_pathways,
                "pathway_fp_combination_count": n_pathways,
                "total_image_count": total_images,
                "image_count_distribution": dist,
                "non_converged_pathway_count": n_not_conv,
                "source_population_counts": src_pop_counts,
                "unsuccessful_count": len(unsuccessful),
                "unsuccessful_status_counts": unsuccessful_status_counts,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 2. Canonical analysis objects
# ─────────────────────────────────────────────────────────────────────────

def build_material_metadata(dft_reference_data):
    """One row per unique icsd_id: CollectionCode, SumFormula. Verbatim port
    of the notebook's original df_ICSD builder (cell 8), reading only the
    canonical DFT reference (no ICSD CSV read)."""
    rows = []
    seen = set()
    for pdata in dft_reference_data["pathways"].values():
        icsd_id = pdata["identifiers"]["icsd_id"]
        if icsd_id in seen:
            continue
        seen.add(icsd_id)
        rows.append({"CollectionCode": icsd_id, "SumFormula": pdata["identifiers"]["sum_formula"]})
    return pd.DataFrame(rows)


def _images_to_atoms(image_dicts, structure_key, energy_key, forces_key):
    atoms_list = []
    for img in image_dicts:
        struct = Structure.from_dict(img[structure_key])
        atoms = _ADAPTOR.get_atoms(struct)
        atoms.calc = SinglePointCalculator(atoms, energy=img[energy_key], forces=img[forces_key])
        atoms_list.append(atoms)
    return atoms_list


def build_dft_neb_image_map(dft_reference_data):
    """Pathway-keyed DFT-NEB images. Returns dft_neb_images_by_path
    ({pathway_key: [ASE Atoms, ...]}) and dft_pathway_keys (list of
    (icsd_str, path_str), same order as dft_reference_data["pathways"])."""
    dft_neb_images_by_path = {}
    dft_pathway_keys = []
    for pkey, pdata in dft_reference_data["pathways"].items():
        icsd_id = pdata["identifiers"]["icsd_id"]
        source_path_id = pdata["identifiers"]["source_path_id"]
        dft_pathway_keys.append((icsd_id, source_path_id))
        dft_neb_images_by_path[pkey] = _images_to_atoms(
            pdata["dft_neb_images"], "structure", "energy_total_eV", "forces_eV_per_angstrom")
    return dft_neb_images_by_path, dft_pathway_keys


def build_full_fp_neb_image_map(fp_results_data, fp_order):
    """Pathway-keyed full FP-NEB images: {fp_key: {pathway_key: [Atoms, ...]}}."""
    models = fp_results_data.get("models", {})
    full_fp_neb_images_by_fp_path = {}
    for fp_key in fp_order:
        full_fp_neb_images_by_fp_path[fp_key] = {}
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            n = len(pdata["final_fp_neb_images"])
            imgs = [pdata["final_fp_neb_images"][str(i)] for i in range(n)]
            full_fp_neb_images_by_fp_path[fp_key][pkey] = _images_to_atoms(
                imgs, "structure", "fp_energy_total_eV", "fp_forces_eV_per_angstrom")
    return full_fp_neb_images_by_fp_path


def build_fp_static_on_dft_neb_image_map(fp_results_data, dft_reference_data, fp_order):
    """Pathway-keyed static FP evaluations on DFT-NEB images:
    {fp_key: {pathway_key: [Atoms, ...]}}. This protocol has no structure of
    its own; each image's structure is paired with the DFT reference
    structure at the same image index (validated equal-length before
    pairing -- see validate_analysis_coverage)."""
    models = fp_results_data.get("models", {})
    fp_static_on_dft_neb_images_by_fp_path = {}
    for fp_key in fp_order:
        fp_static_on_dft_neb_images_by_fp_path[fp_key] = {}
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FP_STATIC_ON_DFT_NEB, {}).get("pathways", {}).items():
            n = len(pdata["images"])
            dft_structs = dft_reference_data["pathways"][pkey]["dft_neb_images"]
            if len(dft_structs) != n:
                raise ValueError(
                    f"Image-count mismatch pairing fp_static_on_dft_neb structures for "
                    f"{fp_key} {pkey}: fp images={n}, DFT reference images={len(dft_structs)}. "
                    f"Refusing to pair with zip() until this is resolved."
                )
            atoms_list = []
            for i in range(n):
                struct = Structure.from_dict(dft_structs[i]["structure"])
                atoms = _ADAPTOR.get_atoms(struct)
                img = pdata["images"][str(i)]
                atoms.calc = SinglePointCalculator(atoms, energy=img["fp_energy_total_eV"], forces=img["fp_forces_eV_per_angstrom"])
                atoms_list.append(atoms)
            fp_static_on_dft_neb_images_by_fp_path[fp_key][pkey] = atoms_list
    return fp_static_on_dft_neb_images_by_fp_path


def build_full_fp_neb_status_map(fp_results_data, fp_order):
    """{fp_key: {(icsd_str, path_str): status_record}} for the full_fp_neb
    protocol only. Static FP evaluations on DFT-NEB images are single-point
    calculations with no NEB optimization step, so there is no equivalent
    status for that protocol -- it is not represented here at all (not
    defaulted, not zero-filled)."""
    models = fp_results_data.get("models", {})
    full_fp_neb_status_by_fp_path = {}
    for fp_key in fp_order:
        full_fp_neb_status_by_fp_path[fp_key] = {}
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            icsd_id = pdata["identifiers"]["icsd_id"]
            source_path_id = pdata["identifiers"]["source_path_id"]
            rec = dict(pdata["convergence"]) if pdata.get("convergence") else {}
            rec.setdefault("neb_converged", pdata["neb_status"]["neb_converged"])
            rec.setdefault("neb_last_fmax", pdata["neb_status"]["neb_last_fmax"])
            rec.setdefault("neb_n_steps", pdata["neb_status"]["neb_n_steps"])
            full_fp_neb_status_by_fp_path[fp_key][(icsd_id, source_path_id)] = rec
    return full_fp_neb_status_by_fp_path


# ─────────────────────────────────────────────────────────────────────────
# 3. Classification (single implementation, unmodified predicate/precedence)
# ─────────────────────────────────────────────────────────────────────────

def classify_energy_profile(images, tolerance=0.01):
    """Classify one NEB pathway's energy profile. Unmodified from the
    notebook's original _classify_consistent (verified byte-identical
    classification counts against the pre-refactor baseline).

    Forward  = char_e - min(E[0], E[-1])
    Backward = char_e - max(E[0], E[-1])
    char_e   = max(intermediates) if above both endpoints, else the
               intermediate with largest deviation from either endpoint.

    Topology priority:
      Abnormal    -- fwd <= 0: char_e below BOTH endpoints (fwd <= 0 implies bwd <= 0).
                     Checked first; endpoint-adjacent dips are expected for
                     barrierless/inverted profiles so Invalid is not applied.
      Invalid     -- fwd > 0 (real hill above lower endpoint) but
                     energies[1] < energies[0] or energies[-2] < energies[-1].
                     (Manuscript label; internal reason retained as
                     "Bad endpoint relaxation".)
      Normal-Hill -- real hill, valid endpoints.
    """
    energies = [atoms.get_potential_energy() for atoms in images]
    intermediates = energies[1:-1]
    E_low = min(energies[0], energies[-1])
    E_high = max(energies[0], energies[-1])

    max_intermed = max(intermediates)
    if max_intermed > E_high:
        char_e = max_intermed
    else:
        char_e = max(intermediates,
                     key=lambda e: max(abs(e - energies[0]), abs(e - energies[-1])))

    fwd = char_e - E_low
    bwd = char_e - E_high
    dE = energies[-1] - energies[0]
    span = max(energies) - min(energies)

    local_maxima = [i for i in range(1, len(energies) - 1)
                    if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]]
    local_minima = [i for i in range(1, len(energies) - 1)
                    if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]]

    if abs(energies[0] - energies[-1]) <= tolerance:
        end_type = "Equal Energy Endpoints"
    elif energies[0] > energies[-1]:
        end_type = "Higher Initial State"
    else:
        end_type = "Higher Final State"

    base = dict(
        energy_forward_barrier=fwd,
        energy_backward_barrier=bwd,
        delta_E=dE,
        energy_range=span,
        n_local_max=len(local_maxima),
        n_local_min=len(local_minima),
        end_type=end_type,
    )

    if fwd <= 0:
        base.update({"Pathway Topology": "Abnormal", "Endpoint Energy Ranking": end_type, "reason": None})
        return base

    if energies[1] < energies[0] or energies[-2] < energies[-1]:
        base.update({"Pathway Topology": "Invalid", "Endpoint Energy Ranking": end_type,
                     "reason": "Bad endpoint relaxation"})
        return base

    base.update({"Pathway Topology": "Normal-Hill", "Endpoint Energy Ranking": end_type, "reason": None})
    return base


def build_dft_path_metrics(dft_neb_images_by_path, dft_pathway_keys):
    """Classify every DFT-NEB pathway. Returns a DataFrame with columns
    ICSD, Path, NEB_Index, Pathway Topology, Endpoint Energy Ranking,
    energy_forward_barrier, energy_backward_barrier, energy_range, delta_E."""
    records = []
    for neb_idx, (icsd_str, path_str) in enumerate(dft_pathway_keys):
        pkey = f"{icsd_str}|{path_str}"
        images = dft_neb_images_by_path[pkey]
        cls = classify_energy_profile(images)
        records.append({
            "ICSD": int(icsd_str), "Path": int(path_str), "NEB_Index": neb_idx,
            "Pathway Topology": cls["Pathway Topology"],
            "Endpoint Energy Ranking": cls["end_type"],
            "energy_forward_barrier": cls["energy_forward_barrier"],
            "energy_backward_barrier": cls["energy_backward_barrier"],
            "energy_range": cls["energy_range"],
            "delta_E": cls["delta_E"],
        })
    return pd.DataFrame(records)


def build_fp_path_metrics(full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path, fp_order):
    """Classify every FP pathway, both protocols. Returns
    fp_path_metrics_by_protocol: {fp_key: {"full_fp_neb": df, "fp_static_on_dft_neb": df}}."""
    sources = {PROTOCOL_FULL_FP_NEB: full_fp_neb_images_by_fp_path,
               PROTOCOL_FP_STATIC_ON_DFT_NEB: fp_static_on_dft_neb_images_by_fp_path}
    fp_path_metrics_by_protocol = {}
    for fp_key in fp_order:
        fp_path_metrics_by_protocol[fp_key] = {}
        for protocol, images_by_fp_path in sources.items():
            path_map = images_by_fp_path.get(fp_key, {})
            if not path_map:
                continue
            records = []
            for neb_idx, (pkey, images) in enumerate(path_map.items()):
                icsd_str, path_str = pkey.split("|")
                cls = classify_energy_profile(images)
                records.append({
                    "ICSD": int(icsd_str), "Path": int(path_str), "NEB_Index": neb_idx,
                    "Pathway Topology": cls["Pathway Topology"],
                    "Endpoint Energy Ranking": cls["end_type"],
                    "Endpoint Energy Asymmetry": cls["end_type"],
                    "energy_forward_barrier": cls["energy_forward_barrier"],
                    "energy_backward_barrier": cls["energy_backward_barrier"],
                    "energy_range": cls["energy_range"],
                    "delta_E": cls["delta_E"],
                })
            fp_path_metrics_by_protocol[fp_key][protocol] = pd.DataFrame(records)
    return fp_path_metrics_by_protocol


# ─────────────────────────────────────────────────────────────────────────
# 4. Endpoint RMSD (Table 9)
# ─────────────────────────────────────────────────────────────────────────

_EP_MATCHER = StructureMatcher(primitive_cell=False, attempt_supercell=False,
                                ltol=0.5, stol=0.5, angle_tol=10.0)


def _compute_endpoint_rmsd_pair(fp_images, dft_images):
    """rmsd/max_dist (actual angstrom) for image 0 and the last image.
    Unmodified from the notebook's original _compute_endpoint_rmsd."""
    results = {}
    last_fp = len(fp_images) - 1
    last_dft = len(dft_images) - 1
    for fp_pos, dft_pos, key in [(0, 0, "img0"), (last_fp, last_dft, "img_last")]:
        try:
            struct_fp = AseAtomsAdaptor.get_structure(fp_images[fp_pos])
            struct_dft = AseAtomsAdaptor.get_structure(dft_images[dft_pos])
            n_atoms = len(struct_fp)
            geom_vol = (struct_fp.volume * struct_dft.volume) ** 0.5
            norm_factor = (geom_vol / n_atoms) ** (1 / 3)
            comp = _EP_MATCHER.get_rms_dist(struct_fp, struct_dft)
            if comp is not None:
                results[f"rmsd_norm_{key}"] = float(comp[0])
                results[f"max_dist_norm_{key}"] = float(comp[1])
                results[f"norm_factor_{key}"] = float(norm_factor)
                results[f"rmsd_{key}"] = float(comp[0]) * norm_factor
                results[f"max_dist_{key}"] = float(comp[1]) * norm_factor
            else:
                results[f"rmsd_norm_{key}"] = float("nan")
                results[f"max_dist_norm_{key}"] = float("nan")
                results[f"norm_factor_{key}"] = float(norm_factor)
                results[f"rmsd_{key}"] = float("nan")
                results[f"max_dist_{key}"] = float("nan")
        except Exception:
            results[f"rmsd_norm_{key}"] = float("nan")
            results[f"max_dist_norm_{key}"] = float("nan")
            results[f"norm_factor_{key}"] = float("nan")
            results[f"rmsd_{key}"] = float("nan")
            results[f"max_dist_{key}"] = float("nan")
    return results


def compute_endpoint_rmsd(fp_path_metrics_by_protocol, dft_neb_images_by_path,
                           full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path,
                           fp_order):
    """{fp_key: {protocol: {pathway_key: rmsd_record}}}. StructureMatcher
    ltol=0.5, stol=0.5, angle_tol=10.0, unchanged from the original notebook."""
    sources = {PROTOCOL_FULL_FP_NEB: full_fp_neb_images_by_fp_path,
               PROTOCOL_FP_STATIC_ON_DFT_NEB: fp_static_on_dft_neb_images_by_fp_path}
    endpoint_rmsd_by_fp_protocol_path = {}
    for fp_key in fp_order:
        endpoint_rmsd_by_fp_protocol_path[fp_key] = {}
        for protocol, images_by_fp_path in sources.items():
            path_map = images_by_fp_path.get(fp_key, {})
            if not path_map or protocol not in fp_path_metrics_by_protocol.get(fp_key, {}):
                continue
            path_rmsd = {}
            for pkey, fp_images in path_map.items():
                dft_images = dft_neb_images_by_path.get(pkey)
                if dft_images is None:
                    continue
                path_rmsd[pkey] = _compute_endpoint_rmsd_pair(fp_images, dft_images)
            endpoint_rmsd_by_fp_protocol_path[fp_key][protocol] = path_rmsd
    return endpoint_rmsd_by_fp_protocol_path


# ─────────────────────────────────────────────────────────────────────────
# 5. Force-error tables (Figure 9, Section 10.6)
# ─────────────────────────────────────────────────────────────────────────

def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def compute_force_angle_error(fp_forces, dft_forces):
    """Per-atom signed Delta|F| (fp magnitude minus dft magnitude), absolute
    |Delta|F||, and Delta-theta (force-angle error, degrees). Returns three
    equal-length lists (one entry per atom)."""
    fp_forces = np.asarray(fp_forces)
    dft_forces = np.asarray(dft_forces)
    n_atoms = len(fp_forces)
    signed_delta_f = [float(np.linalg.norm(fp_forces[i]) - np.linalg.norm(dft_forces[i])) for i in range(n_atoms)]
    abs_delta_f = [abs(v) for v in signed_delta_f]
    delta_theta = [_angle_between(fp_forces[i], dft_forces[i]) for i in range(n_atoms)]
    return signed_delta_f, abs_delta_f, delta_theta


def _safe_nanmean(values):
    """np.nanmean that never warns: returns NaN silently when every entry is
    NaN (e.g. an image where every atom has a zero FP or DFT force vector,
    so the angle is undefined everywhere), instead of letting numpy raise
    'RuntimeWarning: Mean of empty slice'."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _safe_nanmax(values):
    """np.nanmax that never warns: returns NaN silently when every entry is
    NaN, instead of letting numpy raise 'RuntimeWarning: All-NaN slice
    encountered'. Never assigns an artificial zero as a stand-in maximum."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.max(valid)) if valid.size else float("nan")


def _force_error_row(fp_key, icsd_str, path_str, image_index, endpoint_role,
                      fp_energy, fp_forces, dft_energy, dft_forces,
                      protocol, calculation_stage, structure_source,
                      source_population, calculation_status):
    signed_delta_f, abs_delta_f, delta_theta = compute_force_angle_error(fp_forces, dft_forces)
    dft_forces_arr = np.asarray(dft_forces)
    fp_forces_arr = np.asarray(fp_forces)
    n_atoms = len(fp_forces_arr)
    n_valid_angles = int(np.sum(~np.isnan(delta_theta)))
    return {
        "fp_key": fp_key,
        "pathway_key": f"{icsd_str}|{path_str}",
        "icsd_id": icsd_str,
        "source_path_id": path_str,
        "protocol": protocol,
        "calculation_stage": calculation_stage,
        "structure_source": structure_source,
        "image_index": image_index,
        "endpoint_role": endpoint_role,
        "n_atoms": n_atoms,
        "fp_energy_total_eV": fp_energy,
        "dft_energy_total_eV": dft_energy,
        "delta_energy_eV": fp_energy - dft_energy,
        "fp_fmax_eV_per_angstrom": float(np.linalg.norm(fp_forces_arr, axis=1).max()),
        "dft_fmax_eV_per_angstrom": float(np.linalg.norm(dft_forces_arr, axis=1).max()),
        "mean_abs_delta_force_magnitude": _safe_nanmean(abs_delta_f),
        "max_abs_delta_force_magnitude": _safe_nanmax(abs_delta_f),
        "mean_force_angle_error_deg": _safe_nanmean(delta_theta),
        "max_force_angle_error_deg": _safe_nanmax(delta_theta),
        "n_valid_force_angles": n_valid_angles,
        "delta_force_magnitude_migrating_atom": signed_delta_f[MIGRATING_ATOM_INDEX] if n_atoms > MIGRATING_ATOM_INDEX else float("nan"),
        "force_angle_error_migrating_atom_deg": delta_theta[MIGRATING_ATOM_INDEX] if n_atoms > MIGRATING_ATOM_INDEX else float("nan"),
        "source_population": source_population,
        "calculation_status": calculation_status,
        "_signed_delta_force_per_atom": signed_delta_f,
        "_abs_delta_force_per_atom": abs_delta_f,
        "_force_angle_error_per_atom": delta_theta,
        "_dft_force_magnitude_per_atom": [float(np.linalg.norm(f)) for f in dft_forces_arr],
    }


def build_full_fp_neb_path_force_errors(dft_static_on_fp_neb_records_by_fp, fp_order):
    """Force errors on the FP-NEB path (protocol dft_static_on_fp_neb):
    DFT single-point evaluated on the FP's own final full-mode NEB images.
    Limited to the case-study + non-converged-supplemental populations (142
    pathway-FP combinations); NOT all 154 common paths. Returns
    full_fp_neb_path_force_errors_df."""
    rows = []
    for fp_key in fp_order:
        for pkey, pdata in dft_static_on_fp_neb_records_by_fp.get(fp_key, {}).items():
            icsd_id = pdata["identifiers"]["icsd_id"]
            source_path_id = pdata["identifiers"]["source_path_id"]
            source_population = pdata.get("source_population", "unknown")
            n_images = len(pdata["images"])
            for image_index_str, img in pdata["images"].items():
                image_index = int(image_index_str)
                endpoint = ("start" if image_index == 0
                            else "end" if image_index == n_images - 1
                            else f"img{image_index}")
                rows.append(_force_error_row(
                    fp_key, icsd_id, source_path_id, image_index, endpoint,
                    img["fp_energy_total_eV"], img["fp_forces_eV_per_angstrom"],
                    img["dft_energy_total_eV"], img["dft_forces_eV_per_angstrom"],
                    protocol=PROTOCOL_DFT_STATIC_ON_FP_NEB,
                    calculation_stage="dft_single_point_on_fp_full_neb_final_images",
                    structure_source="fp_full_neb_final_structure",
                    source_population=source_population,
                    calculation_status="present",
                ))
    return pd.DataFrame(rows)


def build_dft_neb_path_force_errors(fp_static_on_dft_neb_images_by_fp_path, dft_neb_images_by_path, fp_order):
    """Force errors on the DFT-NEB path (protocol fp_static_on_dft_neb): FP
    single-point evaluated on the DFT-NEB's own image structures. Covers all
    154 common paths (subject to per-FP source coverage). Returns
    dft_neb_path_force_errors_df."""
    rows = []
    for fp_key in fp_order:
        for pkey, fp_atoms_list in fp_static_on_dft_neb_images_by_fp_path.get(fp_key, {}).items():
            icsd_id, source_path_id = pkey.split("|")
            dft_atoms_list = dft_neb_images_by_path[pkey]
            if len(fp_atoms_list) != len(dft_atoms_list):
                raise ValueError(
                    f"Image-count mismatch building dft_neb_path_force_errors for "
                    f"{fp_key} {pkey}: fp={len(fp_atoms_list)} dft={len(dft_atoms_list)}. "
                    f"Refusing to pair with zip() until this is resolved."
                )
            n_images = len(fp_atoms_list)
            for image_index, (fp_atoms, dft_atoms) in enumerate(zip(fp_atoms_list, dft_atoms_list)):
                endpoint = ("start" if image_index == 0
                            else "end" if image_index == n_images - 1
                            else f"img{image_index}")
                rows.append(_force_error_row(
                    fp_key, icsd_id, source_path_id, image_index, endpoint,
                    fp_atoms.get_potential_energy(), fp_atoms.get_forces(),
                    dft_atoms.get_potential_energy(), dft_atoms.get_forces(),
                    protocol=PROTOCOL_FP_STATIC_ON_DFT_NEB,
                    calculation_stage="fp_single_point_on_dft_neb_images",
                    structure_source="dft_neb_reference_structure",
                    source_population=None,
                    calculation_status="present",
                ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 6. Coverage validation (image-pairing safety, Section 16)
# ─────────────────────────────────────────────────────────────────────────

def validate_analysis_coverage(fp_static_on_dft_neb_images_by_fp_path, dft_neb_images_by_path, fp_order):
    """Confirms, for every (fp_key, pathway_key) in fp_static_on_dft_neb,
    that the FP-static image count equals the DFT-reference image count at
    that pathway before any zip()-based pairing occurs. Returns a DataFrame
    of any mismatches found (empty if none). This is a read-only check; it
    never repairs, pads, or truncates data."""
    mismatches = []
    for fp_key in fp_order:
        for pkey, fp_atoms_list in fp_static_on_dft_neb_images_by_fp_path.get(fp_key, {}).items():
            dft_atoms_list = dft_neb_images_by_path.get(pkey)
            if dft_atoms_list is None:
                mismatches.append({"fp_key": fp_key, "pathway_key": pkey,
                                    "fp_image_count": len(fp_atoms_list), "dft_image_count": None,
                                    "issue": "pathway missing from DFT reference"})
                continue
            if len(fp_atoms_list) != len(dft_atoms_list):
                mismatches.append({"fp_key": fp_key, "pathway_key": pkey,
                                    "fp_image_count": len(fp_atoms_list),
                                    "dft_image_count": len(dft_atoms_list),
                                    "issue": "image count mismatch"})
    return pd.DataFrame(mismatches)


# ─────────────────────────────────────────────────────────────────────────
# 7. Barrier-error summaries (Figure 8 / Section 10 numeric tables)
# ─────────────────────────────────────────────────────────────────────────

BARRIER_METRIC_FIELDS = ["energy_forward_barrier", "energy_backward_barrier", "energy_range"]


def compute_barrier_error_summaries(dft_valid_path_metrics_df, fp_path_metrics_by_protocol,
                                     full_fp_neb_status_by_fp_path, endpoint_rmsd_by_fp_protocol_path,
                                     benchmark_pathways_df, fp_order, outlier_threshold,
                                     barrier_metric_fields=BARRIER_METRIC_FIELDS):
    """Verbatim port of the notebook's cells 21+23 (results_full/results_static
    builder + threshold-filtered metrics). One DataFrame per protocol
    (full_fp_neb, fp_static_on_dft_neb), indexed by fp_key, with columns
    MAE_<field>, RMSE_<field>, MAE_<field>_<outlier_threshold,
    RMSE_<field>_<outlier_threshold, mean_RMSD_img0, mean_RMSD_img_last,
    n_total, n_neb_not_conv.

    full_fp_neb: n_neb_not_conv is an int count (from
    full_fp_neb_status_by_fp_path). fp_static_on_dft_neb: single-point
    evaluation has no NEB run, so n_neb_not_conv is the string
    'not applicable' -- never defaulted to 0 or True (spec section 15).

    Also returns full_fp_neb_convergence_summary: {fp_key: {protocol:
    {'n_total', 'n_neb_not_conv'}}} (fp_static_on_dft_neb entries carry
    'not applicable'), used by Figure 8's non-converged-fraction column."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    protocols = (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB)
    metric_records = {p: {} for p in protocols}
    full_fp_neb_convergence_summary = {}

    for fp_key in fp_order:
        full_fp_neb_convergence_summary[fp_key] = {}
        for protocol in protocols:
            if protocol not in fp_path_metrics_by_protocol.get(fp_key, {}):
                continue
            df_m = fp_path_metrics_by_protocol[fp_key][protocol].copy()
            df_m["ICSD"] = df_m["ICSD"].astype(int)
            df_m["Path"] = df_m["Path"].astype(int)

            merged = pd.merge(
                dft_valid_path_metrics_df, df_m, on=["ICSD", "Path"], suffixes=("_DFT", "_FP")
            ).merge(benchmark_pathways_df, on=["ICSD", "Path"], how="inner")
            if merged.empty:
                continue

            X = merged[[f"{m}_FP" for m in barrier_metric_fields]].to_numpy(float)
            merged = merged[np.isfinite(X).all(axis=1) & (X < 1000).all(axis=1) & (X > -1000).all(axis=1)]
            if merged.empty:
                continue

            n_total = len(merged)
            if protocol == PROTOCOL_FULL_FP_NEB:
                clookup = full_fp_neb_status_by_fp_path.get(fp_key, {})
                neb_conv_mask = merged.apply(
                    lambda r: clookup.get((str(int(r["ICSD"])), str(int(r["Path"]))), {})
                              .get("neb_converged", True), axis=1)
                n_neb_not_conv = int((~neb_conv_mask).sum())
                merged_conv = merged[neb_conv_mask]
            else:
                # Static single-point evaluation: no NEB run, so NEB
                # convergence is not applicable -- no filter is applied.
                n_neb_not_conv = "not applicable"
                merged_conv = merged

            full_fp_neb_convergence_summary[fp_key][protocol] = {
                "n_total": n_total, "n_neb_not_conv": n_neb_not_conv,
            }
            if merged_conv.empty:
                continue

            row = {}
            for m in barrier_metric_fields:
                y_true = merged_conv[f"{m}_DFT"]
                y_pred = merged_conv[f"{m}_FP"]
                row[f"MAE_{m}"] = mean_absolute_error(y_true, y_pred)
                row[f"RMSE_{m}"] = np.sqrt(mean_squared_error(y_true, y_pred))
                err = y_pred - y_true
                mask = err.abs() <= outlier_threshold
                if mask.sum() == 0:
                    mae_t = rmse_t = np.nan
                else:
                    mae_t = mean_absolute_error(y_true[mask], y_pred[mask])
                    rmse_t = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                row[f"MAE_{m}_<{outlier_threshold}"] = round(mae_t, 4)
                row[f"RMSE_{m}_<{outlier_threshold}"] = round(rmse_t, 4)

            rmsd_lookup = endpoint_rmsd_by_fp_protocol_path.get(fp_key, {}).get(protocol, {})
            rmsd0_vals, rmsd_last_vals = [], []
            for _, r in merged_conv.iterrows():
                pkey = f"{int(r['ICSD'])}|{int(r['Path'])}"
                rd = rmsd_lookup.get(pkey, {})
                if not np.isnan(rd.get("rmsd_img0", float("nan"))):
                    rmsd0_vals.append(rd["rmsd_img0"])
                if not np.isnan(rd.get("rmsd_img_last", float("nan"))):
                    rmsd_last_vals.append(rd["rmsd_img_last"])
            row["mean_RMSD_img0"] = float(np.mean(rmsd0_vals)) if rmsd0_vals else np.nan
            row["mean_RMSD_img_last"] = float(np.mean(rmsd_last_vals)) if rmsd_last_vals else np.nan
            metric_records[protocol][fp_key] = row

    barrier_error_summary_by_protocol = {
        p: pd.DataFrame.from_dict(metric_records[p], orient="index").round(4) for p in protocols
    }
    return barrier_error_summary_by_protocol, full_fp_neb_convergence_summary


# ─────────────────────────────────────────────────────────────────────────
# 8. Profile summaries: endpoint-energy error, integrated area, and
#    classification agreement (Figure 8 / Section 10 rectangular columns)
# ─────────────────────────────────────────────────────────────────────────

def compute_profile_summaries(fp_path_metrics_by_protocol, dft_path_metrics_df,
                                dft_neb_images_by_path, full_fp_neb_images_by_fp_path,
                                fp_static_on_dft_neb_images_by_fp_path,
                                full_fp_neb_status_by_fp_path, fp_order,
                                area_between_curves, simplify_class):
    """Verbatim port of the notebook's cell 34 (10a: area + classification
    analysis). One shared per-path loop computes, per protocol
    (full_fp_neb, fp_static_on_dft_neb) and per FP: pathway topology
    agreement, endpoint energy-ranking agreement, integrated energy-profile
    area (area_between_curves from neb_plots.py), and endpoint energy-
    difference error -- all four metrics share the same NEB-converged,
    same-topology-valid population, so they are computed together rather
    than via four separate passes over the same paths.

    area_between_curves and simplify_class are passed in (defined in
    neb_plots.py) rather than imported here, keeping plotting-adjacent
    helpers out of this module's own dependency surface.

    Returns (profile_summary_by_protocol, path_records_full): the former is
    {protocol: DataFrame} with one row per FP and columns matching the
    original df_results_full/df_results_static (including the "areas_list"
    column used by the area-error heatmap's threshold-filtered variant);
    the latter is the full_fp_neb per-path record list (area, fwd_err,
    bwd_err, topo_correct, rank_correct, neb_converged) used by the
    non-converged-path diagnostics section.

    "Pathway Topology Accuracy (%)" IS the manuscript's Figure 8
    "Energy-profile shape agr. (%) (full)" metric -- compute_key_neb_metrics_
    summary reads this exact column verbatim (see that function), it is not
    recomputed or re-derived separately. Its population is: DFT topology !=
    "Invalid" (i.e. DFT-Abnormal and DFT-Normal-Hill pathways both included,
    only DFT-Invalid excluded), full_fp_neb NEB-converged, and a finite
    area_between_curves value. There is deliberately only one topology-
    accuracy computation in this module -- if a future change ever needs a
    differently-scoped topology-agreement diagnostic (e.g. restricted to
    DFT-Normal-Hill pathways only, or some other population), it must be
    added as a distinctly-named column/function and never overwrite or get
    conflated with this one, since this one is load-bearing for the
    manuscript figure."""
    images_by_protocol = {
        PROTOCOL_FULL_FP_NEB: full_fp_neb_images_by_fp_path,
        PROTOCOL_FP_STATIC_ON_DFT_NEB: fp_static_on_dft_neb_images_by_fp_path,
    }
    results = {PROTOCOL_FULL_FP_NEB: [], PROTOCOL_FP_STATIC_ON_DFT_NEB: []}
    path_records_full = []

    for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB):
        images_by_fp_path = images_by_protocol[protocol]
        for fp_key in fp_order:
            if protocol not in fp_path_metrics_by_protocol.get(fp_key, {}):
                continue
            df_m = fp_path_metrics_by_protocol[fp_key][protocol].copy()
            df_d = dft_path_metrics_df.copy()
            for df in (df_m, df_d):
                df["ICSD"] = df["ICSD"].astype(str).str.strip()
                df["Path"] = df["Path"].astype(str).str.strip()

            merged = df_m.merge(df_d, on=["ICSD", "Path"], suffixes=("_FP", "_DFT"))
            merged = merged[merged["Pathway Topology_DFT"] != "Invalid"]
            merged = merged[merged["energy_range_FP"].abs() < 1000]

            clookup = full_fp_neb_status_by_fp_path.get(fp_key, {}) if protocol == PROTOCOL_FULL_FP_NEB else {}
            fp_path_map = images_by_fp_path.get(fp_key, {})
            shared_keys = set(dft_neb_images_by_path) & set(fp_path_map)

            all_areas, class_correct_areas, label_correct_areas = [], [], []
            count_class = count_label = n_skipped_nc = 0
            ep_errs_abs, ep_errs_norm, ep_misrank_delta, ep_errs_total = [], [], [], []

            for pkey in shared_keys:
                icsd_str, path_str = pkey.split("|")
                filtered = merged[(merged["ICSD"] == icsd_str) & (merged["Path"] == path_str)]
                if filtered.empty:
                    continue
                row = filtered.iloc[0]

                crec = clookup.get((icsd_str, path_str), {}) if clookup else {}
                neb_ok = crec.get("neb_converged", True) if protocol == PROTOCOL_FULL_FP_NEB else True

                try:
                    area = area_between_curves(dft_neb_images_by_path[pkey], fp_path_map[pkey])
                except Exception:
                    continue
                if not np.isfinite(area):
                    continue

                topo_ok = simplify_class(row["Pathway Topology_DFT"]) == simplify_class(row["Pathway Topology_FP"])
                rank_ok = row["Endpoint Energy Ranking_DFT"] == row["Endpoint Energy Ranking_FP"]

                try:
                    fwd_err = abs(float(row["energy_forward_barrier_FP"]) - float(row["energy_forward_barrier_DFT"]))
                    bwd_err = abs(float(row["energy_backward_barrier_FP"]) - float(row["energy_backward_barrier_DFT"]))
                except Exception:
                    fwd_err = bwd_err = np.nan

                if protocol == PROTOCOL_FULL_FP_NEB:
                    path_records_full.append({
                        "ICSD": icsd_str, "Path": path_str, "fp_key": fp_key,
                        "area": area, "fwd_err": fwd_err, "bwd_err": bwd_err,
                        "topo_correct": topo_ok, "rank_correct": rank_ok,
                        "neb_converged": neb_ok,
                    })

                if not neb_ok:
                    n_skipped_nc += 1
                    continue

                all_areas.append(area)
                try:
                    dft_i = dft_neb_images_by_path[pkey]
                    fp_i = fp_path_map[pkey]
                    na = len(dft_i[0])
                    e0d = dft_i[0].get_potential_energy() / na
                    eNd = dft_i[-1].get_potential_energy() / na
                    e0m = fp_i[0].get_potential_energy() / na
                    eNm = fp_i[-1].get_potential_energy() / na
                    ep_errs_abs.extend([e0m - e0d, eNm - eNd])
                    dE_d = eNd - e0d
                    dE_m = eNm - e0m
                    ep_errs_norm.append(dE_m - dE_d)
                    ep_errs_total.append((dE_m - dE_d) * na)
                    if not rank_ok:
                        ep_misrank_delta.append(abs(dE_d))
                except Exception:
                    pass

                if topo_ok:
                    count_class += 1
                    class_correct_areas.append(area)
                if rank_ok:
                    count_label += 1
                    label_correct_areas.append(area)

            n = len(all_areas)
            tot = sum(all_areas)
            n_tot_all = n + n_skipped_nc

            def safe_mae(vals):
                return float(np.mean(np.abs(vals))) if vals else float("nan")

            def safe_rmse(vals):
                return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")

            results[protocol].append({
                "FP": fp_key,
                "Total Paths (conv)": n,
                "% NEB not conv": round(100 * n_skipped_nc / n_tot_all, 1) if n_tot_all else np.nan,
                "Total Area (eV)": tot,
                "MAE Area (eV)": safe_mae(all_areas),
                "RMSE Area (eV)": safe_rmse(all_areas),
                "Pathway Topology Accuracy (%)": 100 * count_class / n if n else np.nan,
                "Endpoint Energy Ranking Accuracy (%)": 100 * count_label / n if n else np.nan,
                "Topology Area Fraction (%)": 100 * sum(class_correct_areas) / tot if tot else np.nan,
                "Endpoint Ranking Area Fraction (%)": 100 * sum(label_correct_areas) / tot if tot else np.nan,
                "Endpoint E MAE (eV/at)": safe_mae(ep_errs_abs),
                "Endpoint E RMSE (eV/at)": safe_rmse(ep_errs_abs),
                "Endpoint ΔE MAE norm (eV/at)": safe_mae(ep_errs_norm),
                "Endpoint ΔE RMSE norm (eV/at)": safe_rmse(ep_errs_norm),
                "Endpoint ΔE MAE (eV)": safe_mae(ep_errs_total),
                "Endpoint ΔE RMSE (eV)": safe_rmse(ep_errs_total),
                "Misrank ΔE_DFT mean (eV/at)": float(np.mean(ep_misrank_delta)) if ep_misrank_delta else float("nan"),
                "Misrank ΔE_DFT max (eV/at)": float(np.max(ep_misrank_delta)) if ep_misrank_delta else float("nan"),
                "areas_list": list(all_areas),
            })

    profile_summary_by_protocol = {p: pd.DataFrame(results[p]) for p in results}
    return profile_summary_by_protocol, path_records_full


def compute_endpoint_energy_summaries(fp_path_metrics_by_protocol, dft_path_metrics_df,
                                        dft_neb_images_by_path, full_fp_neb_images_by_fp_path,
                                        fp_static_on_dft_neb_images_by_fp_path,
                                        full_fp_neb_status_by_fp_path, fp_order,
                                        area_between_curves, simplify_class):
    """Endpoint energy-ranking agreement and endpoint energy-difference error
    columns from compute_profile_summaries (see that function's docstring
    for why this and compute_integrated_profile_summaries share one
    per-path loop rather than each re-walking the population)."""
    full_df, path_records_full = compute_profile_summaries(
        fp_path_metrics_by_protocol, dft_path_metrics_df, dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path,
        full_fp_neb_status_by_fp_path, fp_order, area_between_curves, simplify_class)
    cols = ["FP", "Total Paths (conv)", "% NEB not conv",
            "Endpoint Energy Ranking Accuracy (%)", "Endpoint Ranking Area Fraction (%)",
            "Endpoint E MAE (eV/at)", "Endpoint E RMSE (eV/at)",
            "Endpoint ΔE MAE norm (eV/at)", "Endpoint ΔE RMSE norm (eV/at)",
            "Endpoint ΔE MAE (eV)", "Endpoint ΔE RMSE (eV)",
            "Misrank ΔE_DFT mean (eV/at)", "Misrank ΔE_DFT max (eV/at)"]
    return {p: df[cols] for p, df in full_df.items()}, path_records_full


def compute_integrated_profile_summaries(fp_path_metrics_by_protocol, dft_path_metrics_df,
                                           dft_neb_images_by_path, full_fp_neb_images_by_fp_path,
                                           fp_static_on_dft_neb_images_by_fp_path,
                                           full_fp_neb_status_by_fp_path, fp_order,
                                           area_between_curves, simplify_class):
    """Integrated energy-profile-difference (area) and pathway-topology
    agreement columns from compute_profile_summaries (see that function's
    docstring)."""
    full_df, _ = compute_profile_summaries(
        fp_path_metrics_by_protocol, dft_path_metrics_df, dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path,
        full_fp_neb_status_by_fp_path, fp_order, area_between_curves, simplify_class)
    cols = ["FP", "Total Paths (conv)", "% NEB not conv", "Total Area (eV)",
            "MAE Area (eV)", "RMSE Area (eV)",
            "Pathway Topology Accuracy (%)", "Topology Area Fraction (%)", "areas_list"]
    return {p: df[cols] for p, df in full_df.items()}


def compute_key_neb_metrics_summary(dft_valid_path_metrics_df, fp_path_metrics_by_protocol,
                                     dft_path_metrics_df, full_fp_neb_status_by_fp_path,
                                     endpoint_rmsd_by_fp_protocol_path, benchmark_pathways_df,
                                     dft_neb_images_by_path, full_fp_neb_images_by_fp_path,
                                     fp_static_on_dft_neb_images_by_fp_path,
                                     fp_order, outlier_threshold, area_between_curves, simplify_class):
    """Compact key-NEB-metrics summary (manuscript Figure 8). Reuses
    compute_barrier_error_summaries and compute_profile_summaries internally
    for the per-path forward/backward barrier and profile calculations (same
    populations, denominators, and formulas as those functions already
    verified), but only exposes the six columns the compact summary needs --
    it does not compute or expose separate forward/backward MAE/RMSE,
    outlier-threshold-restricted errors, energy-range MAE/RMSE, "% NEB not
    conv" as a numeric column, or mean image-0/image-N RMSD; those remain
    available from compute_barrier_error_summaries/compute_profile_summaries
    directly for callers that need them.

    Returns key_neb_metrics_summary_df, indexed by fp_key, with columns:
      "Non-conv. paths (n/total)", "Barrier error (eV) (full)",
      "Barrier error (eV) (static)", "Endpoint energy-diff. error (eV) (full)",
      "Endpoint energy rank. agr. (%) (full)", "Energy-profile shape agr. (%) (full)"."""
    barrier_error_summary_by_protocol, full_fp_neb_convergence_summary = compute_barrier_error_summaries(
        dft_valid_path_metrics_df, fp_path_metrics_by_protocol, full_fp_neb_status_by_fp_path,
        endpoint_rmsd_by_fp_protocol_path, benchmark_pathways_df, fp_order, outlier_threshold,
    )
    full_barrier_df = barrier_error_summary_by_protocol[PROTOCOL_FULL_FP_NEB]
    static_barrier_df = barrier_error_summary_by_protocol[PROTOCOL_FP_STATIC_ON_DFT_NEB]

    profile_summary_by_protocol, _ = compute_profile_summaries(
        fp_path_metrics_by_protocol, dft_path_metrics_df, dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path,
        full_fp_neb_status_by_fp_path, fp_order, area_between_curves, simplify_class,
    )
    full_profile_df = profile_summary_by_protocol[PROTOCOL_FULL_FP_NEB].set_index("FP")

    def _combined_barrier(df):
        # Matches the original notebook's cell 24 -> cell 37 order of operations:
        # forward/backward MAE and RMSE are rounded to 2 decimal places FIRST
        # (cell 24's df_full_cleaned = df_full...round(2)), and only then
        # combined (cell 37). Combining at full precision and rounding only
        # for display gives a numerically different (if arguably "cleaner")
        # result that does not match the manuscript-verified values, so the
        # round-then-combine order must be preserved exactly here.
        fwd_mae = df["MAE_energy_forward_barrier"].round(2)
        bwd_mae = df["MAE_energy_backward_barrier"].round(2)
        fwd_rmse = df["RMSE_energy_forward_barrier"].round(2)
        bwd_rmse = df["RMSE_energy_backward_barrier"].round(2)
        mae = (fwd_mae + bwd_mae) / 2
        rmse = np.sqrt((fwd_rmse ** 2 + bwd_rmse ** 2) / 2)
        return mae.map(lambda x: f"{x:.4f}") + " / " + rmse.map(lambda x: f"{x:.4f}")

    index = full_barrier_df.index
    non_conv_labels = []
    for fp_key in index:
        s = full_fp_neb_convergence_summary.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {})
        n_not, n_tot = s.get("n_neb_not_conv", 0), s.get("n_total", 0)
        non_conv_labels.append(f"{n_not}/{n_tot}" if n_tot else "—")

    ep_dE_mae = full_profile_df["Endpoint ΔE MAE (eV)"].reindex(index)
    ep_dE_rmse = full_profile_df["Endpoint ΔE RMSE (eV)"].reindex(index)
    ep_dE_tri = (ep_dE_mae.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
                 + " / " + ep_dE_rmse.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))

    key_neb_metrics_summary_df = pd.DataFrame({
        "Non-conv. paths (n/total)": non_conv_labels,
        "Barrier error (eV) (full)": _combined_barrier(full_barrier_df).values,
        "Barrier error (eV) (static)": _combined_barrier(static_barrier_df).reindex(index).values,
        "Endpoint energy-diff. error (eV) (full)": ep_dE_tri.values,
        "Endpoint energy rank. agr. (%) (full)": full_profile_df["Endpoint Energy Ranking Accuracy (%)"].reindex(index).values,
        "Energy-profile shape agr. (%) (full)": full_profile_df["Pathway Topology Accuracy (%)"].reindex(index).values,
    }, index=index)
    return key_neb_metrics_summary_df


# ─────────────────────────────────────────────────────────────────────────
# 9. Threshold-case membership (Section 12.1-12.4 diagnostics)
# ─────────────────────────────────────────────────────────────────────────

def _endpoint_rmsd_barrier_status(icsd, path_id, endpoint_rmsd_by_fp_protocol_path,
                                    dft_valid_path_metrics_df, fp_path_metrics_by_protocol,
                                    fp_key, rmsd_op, rmsd_val, barrier_op, barrier_val):
    """One row's worth of check_thresholds' per-path RMSD/barrier status
    (BOTH/EXACTLY ONE/NEITHER/no data), verbatim from the notebook's
    check_thresholds, for the full_fp_neb protocol."""
    key = (str(icsd), str(path_id))
    rd = endpoint_rmsd_by_fp_protocol_path.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get(f"{icsd}|{path_id}", {})
    r0 = rd.get("rmsd_img0", float("nan"))
    rN = rd.get("rmsd_img_last", float("nan"))

    def ok_rmsd(v):
        return (v < rmsd_val) if rmsd_op == "<" else (v > rmsd_val)

    s0 = ok_rmsd(r0) if not np.isnan(r0) else None
    sN = ok_rmsd(rN) if not np.isnan(rN) else None
    if s0 is None and sN is None:
        rmsd_status = "no data"
    elif s0 and sN:
        rmsd_status = "BOTH"
    elif s0 or sN:
        rmsd_status = "EXACTLY ONE"
    else:
        rmsd_status = "NEITHER"

    fwd_err = bwd_err = float("nan")
    dv = dft_valid_path_metrics_df
    row_dv = dv[(dv["ICSD"].astype(str) == str(icsd)) & (dv["Path"].astype(str) == str(path_id))]
    if not row_dv.empty and PROTOCOL_FULL_FP_NEB in fp_path_metrics_by_protocol.get(fp_key, {}):
        dm = fp_path_metrics_by_protocol[fp_key][PROTOCOL_FULL_FP_NEB]
        row_m = dm[(dm["ICSD"].astype(str) == str(icsd)) & (dm["Path"].astype(str) == str(path_id))]
        if not row_m.empty:
            fwd_err = abs(row_m.iloc[0]["energy_forward_barrier"] - row_dv.iloc[0]["energy_forward_barrier"])
            bwd_err = abs(row_m.iloc[0]["energy_backward_barrier"] - row_dv.iloc[0]["energy_backward_barrier"])

    def ok_barrier(v):
        return (v < barrier_val) if barrier_op == "<" else (v > barrier_val)

    bf = ok_barrier(fwd_err) if not np.isnan(fwd_err) else None
    bb = ok_barrier(bwd_err) if not np.isnan(bwd_err) else None
    if bf is None and bb is None:
        barrier_status = "no data"
    elif bf and bb:
        barrier_status = "BOTH"
    elif bf or bb:
        barrier_status = "EXACTLY ONE"
    else:
        barrier_status = "NEITHER"

    return {
        "fp_key": fp_key, "ICSD": icsd, "path_id": path_id,
        "rmsd_img0": round(r0, 4) if not np.isnan(r0) else np.nan,
        "rmsd_imgN": round(rN, 4) if not np.isnan(rN) else np.nan,
        "rmsd_status": rmsd_status,
        "fwd_err": round(fwd_err, 4) if not np.isnan(fwd_err) else np.nan,
        "bwd_err": round(bwd_err, 4) if not np.isnan(bwd_err) else np.nan,
        "barrier_status": barrier_status,
    }


def _mode_ok(status, mode):
    if mode == "both":
        return status == "BOTH"
    return status in ("BOTH", "EXACTLY ONE")


def build_threshold_case_membership(case_thresholds, full_fp_neb_path_force_errors_df,
                                      endpoint_rmsd_by_fp_protocol_path, dft_valid_path_metrics_df,
                                      fp_path_metrics_by_protocol, full_fp_neb_status_by_fp_path,
                                      fp_order):
    """Verbatim port of the notebook's Section A cells (check_thresholds +
    CASE_THRESHOLDS classification, cells 55/57). Classification universe:
    every (fp_key, ICSD, path) in full_fp_neb_path_force_errors_df (the
    dft_static_on_fp_neb, case-study + non-converged-supplemental
    population) whose full-mode NEB run converged. Every such path is
    classified against case_thresholds' "both" vs "at least one" semantics
    for RMSD and barrier error independently.

    Returns (case_status_by_case, case_force_errors_by_case):
    case_status_by_case[case_name] is the qualifying-paths status
    DataFrame (rmsd_status/barrier_status columns); case_force_errors_by_case[case_name]
    is full_fp_neb_path_force_errors_df filtered down to every image
    belonging to a qualifying path (the force-error averaging population
    for that case)."""
    all_targets = {}
    for fp_key, grp in full_fp_neb_path_force_errors_df.groupby("fp_key"):
        clookup = full_fp_neb_status_by_fp_path.get(fp_key, {})
        candidates = grp[["icsd_id", "source_path_id"]].drop_duplicates().apply(tuple, axis=1).tolist()
        converged = [
            (icsd, pid) for icsd, pid in candidates
            if clookup.get((str(icsd), str(pid)), {}).get("neb_converged", True)
        ]
        if converged:
            all_targets[fp_key] = converged

    case_status_by_case, case_force_errors_by_case = {}, {}
    for case_name, cfg in case_thresholds.items():
        rows = []
        for fp_key, paths in all_targets.items():
            for icsd, pid in paths:
                rows.append(_endpoint_rmsd_barrier_status(
                    icsd, pid, endpoint_rmsd_by_fp_protocol_path, dft_valid_path_metrics_df,
                    fp_path_metrics_by_protocol, fp_key,
                    cfg["rmsd_op"], cfg["rmsd_val"], cfg["barrier_op"], cfg["barrier_val"]))
        status_df = pd.DataFrame(rows)
        if status_df.empty:
            case_status_by_case[case_name] = status_df
            case_force_errors_by_case[case_name] = full_fp_neb_path_force_errors_df.iloc[0:0]
            continue

        rmsd_ok = status_df["rmsd_status"].apply(lambda s: _mode_ok(s, cfg["rmsd_mode"]))
        barrier_ok = status_df["barrier_status"].apply(lambda s: _mode_ok(s, cfg["barrier_mode"]))
        qualifying = status_df[rmsd_ok & barrier_ok].reset_index(drop=True)
        case_status_by_case[case_name] = qualifying

        if qualifying.empty:
            case_force_errors_by_case[case_name] = full_fp_neb_path_force_errors_df.iloc[0:0]
            continue
        keys = qualifying[["fp_key", "ICSD", "path_id"]].drop_duplicates().copy()
        keys.columns = ["fp_key", "icsd_id", "source_path_id"]
        keys["icsd_id"] = keys["icsd_id"].astype(str)
        keys["source_path_id"] = keys["source_path_id"].astype(str)
        da = full_fp_neb_path_force_errors_df.copy()
        da["icsd_id"] = da["icsd_id"].astype(str)
        da["source_path_id"] = da["source_path_id"].astype(str)
        case_force_errors_by_case[case_name] = da.merge(
            keys, on=["fp_key", "icsd_id", "source_path_id"], how="inner")

    return case_status_by_case, case_force_errors_by_case


def build_static_barrier_case_membership(fp_path_metrics_by_protocol, dft_path_metrics_df,
                                           dft_valid_path_metrics_df, benchmark_pathways_df,
                                           fp_order, good_threshold_ev=0.05, bad_threshold_ev=0.10):
    """Verbatim port of the notebook's "13a: Barrier-error case classification,
    static mode" cell. The FP is evaluated on the DFT-finalized NEB images
    (fp_static_on_dft_neb): structures are identical to DFT, so no endpoint
    RMSD comparison applies -- cases are barrier-error-only. This protocol
    has no NEB run (see compute_barrier_error_summaries), so there is no
    convergence filter here either.

    Case A (good): both fwd_err and bwd_err < good_threshold_ev.
    Case B (poor): either fwd_err or bwd_err > bad_threshold_ev.
    Moderate: falls in neither case.

    Note: good_threshold_ev (0.05 eV here) is a distinct, independently
    retained threshold from the 0.01 eV "good" barrier threshold used in
    case 1-b of build_threshold_case_membership -- these are two separate
    diagnostics with two separate thresholds by original design, not an
    inconsistency to silently resolve (spec section 18).

    Returns static_barrier_cases_df with one row per (fp_key, ICSD, Path)."""
    rows = []
    for fp_key in fp_order:
        if PROTOCOL_FP_STATIC_ON_DFT_NEB not in fp_path_metrics_by_protocol.get(fp_key, {}):
            continue
        df_m = fp_path_metrics_by_protocol[fp_key][PROTOCOL_FP_STATIC_ON_DFT_NEB].copy()
        for col in ("ICSD", "Path"):
            df_m[col] = df_m[col].astype(int)
        merged = pd.merge(
            dft_valid_path_metrics_df, df_m, on=["ICSD", "Path"], suffixes=("_DFT", "_FP")
        ).merge(benchmark_pathways_df, on=["ICSD", "Path"], how="inner")
        if merged.empty:
            continue
        X = merged[["energy_forward_barrier_FP", "energy_backward_barrier_FP"]].to_numpy(float)
        merged = merged[np.isfinite(X).all(axis=1) & (X < 1000).all(axis=1) & (X > -1000).all(axis=1)]

        for _, r in merged.iterrows():
            fwd_err = abs(r["energy_forward_barrier_FP"] - r["energy_forward_barrier_DFT"])
            bwd_err = abs(r["energy_backward_barrier_FP"] - r["energy_backward_barrier_DFT"])
            rows.append({
                "FP": fp_key, "ICSD": int(r["ICSD"]), "Path": int(r["Path"]),
                "fwd_DFT (eV)": round(r["energy_forward_barrier_DFT"], 3),
                "bwd_DFT (eV)": round(r["energy_backward_barrier_DFT"], 3),
                "fwd_FP (eV)": round(r["energy_forward_barrier_FP"], 3),
                "bwd_FP (eV)": round(r["energy_backward_barrier_FP"], 3),
                "fwd_err (eV)": round(fwd_err, 3),
                "bwd_err (eV)": round(bwd_err, 3),
                "topology_DFT": r.get("Pathway Topology_DFT", "—"),
                "topology_FP": r.get("Pathway Topology_FP", "—"),
                "delta_E_DFT (eV)": round(r.get("delta_E_DFT", float("nan")), 3),
            })

    static_barrier_cases_df = pd.DataFrame(rows)
    if static_barrier_cases_df.empty:
        return static_barrier_cases_df
    case_a = ((static_barrier_cases_df["fwd_err (eV)"] < good_threshold_ev) &
              (static_barrier_cases_df["bwd_err (eV)"] < good_threshold_ev))
    case_b = ((static_barrier_cases_df["fwd_err (eV)"] > bad_threshold_ev) |
              (static_barrier_cases_df["bwd_err (eV)"] > bad_threshold_ev))
    static_barrier_cases_df["case"] = np.select(
        [case_a, case_b], ["A (good)", "B (poor)"], default="Moderate")
    return static_barrier_cases_df


# ─────────────────────────────────────────────────────────────────────────
# 10. Reusable public API for external datasets
#
# Everything above this point was written against, and is verified against,
# the FPBench 154-path benchmark. The functions below add nothing new
# scientifically -- they orchestrate the existing functions above (same
# formulas, same thresholds, same denominators, same averaging conventions)
# behind one public entry point that does not assume a 154-path dataset,
# plus general-purpose input validation and a documented example for anyone
# supplying their own DFT reference + FP results in the same JSON schema.
#
# The FPBench-specific validate_neb_datasets(...) (with its 154-path
# expected_pathway_count check) is unchanged and still what the benchmark
# notebook itself calls; validate_neb_analysis_inputs(...) below is the
# general-purpose sibling used by build_neb_analysis_results(...) and by
# anyone validating a non-benchmark dataset directly.
# ─────────────────────────────────────────────────────────────────────────


def validate_neb_analysis_inputs(reference_data, fp_results, expected_pathways=None, fp_order=None):
    """General-purpose validator for the two-dictionary NEB analysis input
    schema (DFT reference + FP results), usable on any dataset following the
    canonical JSON schema documented in input_data/README.md and
    results/README.md (and, briefly, in the analysis notebook's "Using
    FPBench data or another NEB dataset" section) -- not only the FPBench
    154-path benchmark. `expected_pathways` is optional and, when given, is only
    checked as an informational count; no other check requires it or any
    other fixed pathway count.

    Never discards, repairs, or silently reinterprets data. Every issue
    becomes one row in the returned report; missing/failed/non-converged
    records are expected to exist in the input and are checked for internal
    consistency, not filtered out here (eligibility for a given metric is
    decided later, by the existing manuscript metric functions).

    Checks performed (see the returned report's `checks` list / `summary_df`
    for the exact per-check detail):
      - unique DFT-reference pathway keys, keys consistent with their own
        identifiers.icsd_id/identifiers.source_path_id;
      - (optional) DFT-reference pathway count vs expected_pathways;
      - DFT-reference image_index sequences are 0..N-1 in order;
      - DFT-reference first/last images are tagged endpoint_role
        "initial"/"final";
      - structure/forces array-length consistency (forces count == site
        count) and atom-count consistency across a pathway's own images;
      - every fp_order key present in fp_results["models"];
      - protocol-schema shape checks that catch protocol mixing directly:
        full_fp_neb images must carry their own "structure" (they are FP-
        relaxed, not reused from DFT); fp_static_on_dft_neb images must NOT
        carry a "structure" (this protocol reuses the DFT-NEB reference
        structure at the same image index, so it should never appear to
        have its own); dft_static_on_fp_neb images must carry
        fp_structure/fp_energy_total_eV/fp_forces_eV_per_angstrom/
        dft_energy_total_eV/dft_forces_eV_per_angstrom;
      - full_fp_neb/fp_static_on_dft_neb pathway keys exist in the DFT
        reference (no orphan FP-side pathway keys);
      - fp_static_on_dft_neb image counts match their DFT-NEB pathway's
        image count (the same check validate_analysis_coverage performs,
        run here proactively before any zip()-based pairing is attempted
        downstream);
      - dft_static_on_fp_neb image counts do not exceed the corresponding
        full_fp_neb pathway's image count, where both are present;
      - no duplicate pathway keys within any single protocol branch;
      - full_fp_neb pathways carry an explicit neb_status.neb_converged
        value (never defaulted).

    Not attempted: automated detection of "wrong units" (e.g. eV vs
    eV/atom) from value magnitude alone. The canonical schema declares
    units in a top-level "units" dict (checked for presence/expected string
    here); distinguishing merely-implausible-looking-but-correct physics
    from genuinely mislabeled units cannot be done reliably without domain-
    specific thresholds, so it is intentionally not attempted -- callers
    supplying their own data are responsible for ensuring energies are in
    eV and forces in eV/angstrom, as declared.

    Returns an object with .summary_df (DataFrame, one row per check),
    .ok (bool, True iff every check passed), and .checks (the same data as
    a plain list of dicts)."""
    checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    try:
        reference_data = _normalize_reference_data(reference_data)
        record("dft_neb_reference/dft_neb_images normalization", True)
    except ValueError as e:
        record("dft_neb_reference/dft_neb_images normalization", False, str(e))

    pathways = reference_data.get("pathways", {})
    models = fp_results.get("models", {})
    if fp_order is None:
        fp_order = list(models.keys())

    # -- standardized-file required metadata (schema_version/component/
    #    dataset_name/units on both sides; see input_data/README.md
    #    "Required vs optional fields" for the full required-field list).
    #    Informational for analysis (build_neb_analysis_results never reads
    #    these), but required for a standardized publishable file per spec
    #    section F -- checked here so a publisher gets one clear report.
    for label, doc in (("DFT reference", reference_data), ("FP results", fp_results)):
        for field in ("schema_version", "component", "dataset_name"):
            record(f"{label} declares '{field}'", field in doc and doc.get(field) not in (None, ""),
                   doc.get(field))

    # -- declared units, both files, checked separately, plus cross-file agreement --
    units = reference_data.get("units", {})
    fp_units = fp_results.get("units", {})
    record("DFT reference declares units.energy == 'eV'", units.get("energy") == "eV", units.get("energy"))
    record("DFT reference declares units.forces == 'eV/angstrom'",
           units.get("forces") == "eV/angstrom", units.get("forces"))
    record("FP results declares units.energy == 'eV'", fp_units.get("energy") == "eV", fp_units.get("energy"))
    record("FP results declares units.forces == 'eV/angstrom'",
           fp_units.get("forces") == "eV/angstrom", fp_units.get("forces"))
    record("DFT reference and FP results agree on units.energy",
           units.get("energy") == fp_units.get("energy"),
           f"reference={units.get('energy')!r} fp_results={fp_units.get('energy')!r}")
    record("DFT reference and FP results agree on units.forces",
           units.get("forces") == fp_units.get("forces"),
           f"reference={units.get('forces')!r} fp_results={fp_units.get('forces')!r}")
    # Numerical magnitude alone cannot prove a user mislabeled units (e.g. eV
    # vs eV/atom look identical in shape) -- intentionally not attempted, see
    # this function's docstring "Not attempted" note. Only declared-string
    # agreement is checked above.

    # -- unique pathway identifiers, key/identifier consistency --
    seen_keys = set()
    dup_keys = []
    bad_key_identifier = []
    for pkey, pdata in pathways.items():
        ident = pdata.get("identifiers", {})
        icsd, path = ident.get("icsd_id"), ident.get("source_path_id")
        if f"{icsd}|{path}" != pkey:
            bad_key_identifier.append(pkey)
        if pkey in seen_keys:
            dup_keys.append(pkey)
        seen_keys.add(pkey)
    record("DFT reference pathway keys are unique", len(dup_keys) == 0, dup_keys[:5])
    record("DFT reference pathway keys match their own identifiers",
           len(bad_key_identifier) == 0, bad_key_identifier[:5])

    if expected_pathways is not None:
        record(f"DFT reference pathway count == {expected_pathways}",
               len(pathways) == expected_pathways, f"found {len(pathways)}")

    # -- image ordering and endpoint tags --
    bad_image_order, bad_endpoints = [], []
    for pkey, pdata in pathways.items():
        imgs = pdata.get("dft_neb_images", [])
        if [im.get("image_index") for im in imgs] != list(range(len(imgs))):
            bad_image_order.append(pkey)
        if len(imgs) >= 2:
            if imgs[0].get("endpoint_role") != "initial":
                bad_endpoints.append((pkey, "first image not tagged 'initial'"))
            if imgs[-1].get("endpoint_role") != "final":
                bad_endpoints.append((pkey, "last image not tagged 'final'"))
    record("DFT reference image_index sequences are 0..N-1 in order",
           len(bad_image_order) == 0, bad_image_order[:5])
    record("DFT reference first/last images tagged initial/final",
           len(bad_endpoints) == 0, bad_endpoints[:5])

    # -- structure/energy/force array dimensions, atom-count consistency --
    dim_issues = []
    for pkey, pdata in pathways.items():
        atom_counts = set()
        for im in pdata.get("dft_neb_images", []):
            sites = im.get("structure", {}).get("sites", [])
            forces = im.get("forces_eV_per_angstrom", [])
            if len(forces) != len(sites):
                dim_issues.append((pkey, im.get("image_index"), "forces/sites length mismatch"))
            atom_counts.add(len(sites))
            if not isinstance(im.get("energy_total_eV"), (int, float)):
                dim_issues.append((pkey, im.get("image_index"), "energy_total_eV not numeric"))
        if len(atom_counts) > 1:
            dim_issues.append((pkey, None, f"atom count varies across images: {sorted(atom_counts)}"))
    record("DFT reference force-array/structure dimensions and atom counts consistent",
           len(dim_issues) == 0, dim_issues[:5])

    # -- FP coverage --
    record("every fp_order key present in fp_results['models']",
           set(fp_order) <= set(models.keys()), sorted(set(fp_order) - set(models.keys())))

    # -- protocol-schema shape checks (direct protocol-mixing detection) --
    protocol_shape_issues = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for pkey, pdata in model.get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            for ik, img in pdata.get("final_fp_neb_images", {}).items():
                if "structure" not in img:
                    protocol_shape_issues.append(
                        (fp_key, PROTOCOL_FULL_FP_NEB, pkey, ik,
                         "missing its own 'structure' -- full_fp_neb images must be FP-relaxed, "
                         "not reused from another protocol"))
        for pkey, pdata in model.get(PROTOCOL_FP_STATIC_ON_DFT_NEB, {}).get("pathways", {}).items():
            for ik, img in pdata.get("images", {}).items():
                if "structure" in img:
                    protocol_shape_issues.append(
                        (fp_key, PROTOCOL_FP_STATIC_ON_DFT_NEB, pkey, ik,
                         "unexpectedly carries its own 'structure' -- this protocol must reuse the "
                         "DFT-NEB reference structure at the same image index, never its own"))
        for pkey, pdata in model.get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("pathways", {}).items():
            for ik, img in pdata.get("images", {}).items():
                missing = [f for f in ("fp_structure", "fp_energy_total_eV", "fp_forces_eV_per_angstrom",
                                        "dft_energy_total_eV", "dft_forces_eV_per_angstrom") if f not in img]
                if missing:
                    protocol_shape_issues.append((fp_key, PROTOCOL_DFT_STATIC_ON_FP_NEB, pkey, ik,
                                                   f"missing fields: {missing}"))
    record("protocol-specific image schemas are not mixed (structure/field-presence check)",
           len(protocol_shape_issues) == 0, protocol_shape_issues[:5])

    # -- FP <-> DFT pathway-key correspondence --
    orphan_fp_pathways = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB):
            for pkey in model.get(protocol, {}).get("pathways", {}):
                if pkey not in pathways:
                    orphan_fp_pathways.append((fp_key, protocol, pkey))
    record("full_fp_neb/fp_static_on_dft_neb pathway keys exist in the DFT reference",
           len(orphan_fp_pathways) == 0, orphan_fp_pathways[:5])

    # -- fp_static_on_dft_neb <-> DFT-NEB image-count correspondence --
    static_mismatches = []
    for fp_key in fp_order:
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FP_STATIC_ON_DFT_NEB, {}).get("pathways", {}).items():
            dft_pdata = pathways.get(pkey)
            if dft_pdata is None:
                continue
            n_fp, n_dft = len(pdata.get("images", {})), len(dft_pdata.get("dft_neb_images", []))
            if n_fp != n_dft:
                static_mismatches.append((fp_key, pkey, n_fp, n_dft))
    record("fp_static_on_dft_neb image counts match their DFT-NEB pathway (pre-zip() safety check)",
           len(static_mismatches) == 0, static_mismatches[:5])

    # -- fp_static_on_dft_neb: exact image-INDEX-SET correspondence, not just
    #    counts (a missing index 2 and a duplicated index 1 have the same
    #    count as the correct set, but pair against the wrong DFT image if
    #    only counted) --
    static_index_set_mismatches = []
    for fp_key in fp_order:
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FP_STATIC_ON_DFT_NEB, {}).get("pathways", {}).items():
            dft_pdata = pathways.get(pkey)
            if dft_pdata is None:
                continue
            fp_indices = set(pdata.get("images", {}).keys())
            expected_indices = {str(i) for i in range(len(dft_pdata.get("dft_neb_images", [])))}
            if fp_indices != expected_indices:
                static_index_set_mismatches.append(
                    (fp_key, pkey, "missing=" + str(sorted(expected_indices - fp_indices)),
                     "unexpected=" + str(sorted(fp_indices - expected_indices))))
    record("fp_static_on_dft_neb image index sets exactly match their DFT-NEB pathway "
           "(not just counts; rejects missing/duplicated/unexpected indices)",
           len(static_index_set_mismatches) == 0, static_index_set_mismatches[:5])

    # -- dft_static_on_fp_neb: structural correspondence to the SAME FP's own
    #    full_fp_neb image at the same index (species order + atom count
    #    exact match; deliberately no StructureMatcher-style reordering
    #    tolerance here -- force arrays are only meaningful paired against
    #    the exact atom order they were computed on) --
    dft_on_fp_structure_mismatches = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        full_pathways = model.get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {})
        for pkey, pdata in model.get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("pathways", {}).items():
            full_pdata = full_pathways.get(pkey)
            if full_pdata is None:
                continue
            full_images = full_pdata.get("final_fp_neb_images", {})
            for ik, img in pdata.get("images", {}).items():
                full_img = full_images.get(ik)
                if full_img is None:
                    dft_on_fp_structure_mismatches.append(
                        (fp_key, pkey, ik, "no full_fp_neb image at this index to compare against"))
                    continue
                diag_sites = img.get("fp_structure", {}).get("sites", [])
                full_sites = full_img.get("structure", {}).get("sites", [])
                if len(diag_sites) != len(full_sites):
                    dft_on_fp_structure_mismatches.append(
                        (fp_key, pkey, ik, f"atom count mismatch: diag={len(diag_sites)} full_fp_neb={len(full_sites)}"))
                    continue
                diag_species = [s.get("species", [{}])[0].get("element") for s in diag_sites]
                full_species = [s.get("species", [{}])[0].get("element") for s in full_sites]
                if diag_species != full_species:
                    dft_on_fp_structure_mismatches.append(
                        (fp_key, pkey, ik, "species order mismatch (StructureMatcher-style "
                                           "reordering is deliberately not accepted here)"))
                    continue
                diag_xyz = np.array([s.get("xyz", []) for s in diag_sites], dtype=float)
                full_xyz = np.array([s.get("xyz", []) for s in full_sites], dtype=float)
                # atol=1e-6 A: real FPBench data duplicates these coordinates
                # to ~1e-15 A (float64 machine epsilon, confirmed by direct
                # inspection of all ~95k real site-pairs) since both branches
                # store the exact same structure, not an independently-relaxed
                # copy. 1e-6 A is far above that noise floor and far below any
                # physically distinguishable atomic position (~1e-3 A), so a
                # genuine wrong-pathway/wrong-image substitution is still
                # caught while true float round-trip noise never is.
                if diag_xyz.shape != full_xyz.shape or not np.allclose(diag_xyz, full_xyz, rtol=0, atol=1e-6):
                    dft_on_fp_structure_mismatches.append(
                        (fp_key, pkey, ik, "duplicated fp_structure site coordinates differ from full_fp_neb's "
                                           "own record by more than 1e-6 A (not an independent relaxation, so this "
                                           "should be an exact duplicate)"))
                    continue
                diag_forces = img.get("fp_forces_eV_per_angstrom")
                full_forces = full_img.get("fp_forces_eV_per_angstrom")
                if diag_forces is not None and full_forces is not None:
                    if len(diag_forces) != len(full_forces):
                        dft_on_fp_structure_mismatches.append(
                            (fp_key, pkey, ik, "duplicated fp_forces array shape mismatch vs full_fp_neb"))
                    # atol=5e-4 eV/A: real FPBench data has up to ~1.5e-4
                    # eV/A float32-vs-float64 serialization noise between the
                    # two branches' independently-stored copies (confirmed by
                    # direct inspection, exact-power-of-2-sized diffs -- a
                    # quantization artifact, not a real discrepancy). Well
                    # above that noise floor, still 10-1000x below any
                    # physically meaningful force difference anywhere in this
                    # analysis (the smallest scientifically-relevant force
                    # error scale used elsewhere is ~0.001 eV/A).
                    elif not np.allclose(diag_forces, full_forces, rtol=0, atol=5e-4):
                        dft_on_fp_structure_mismatches.append(
                            (fp_key, pkey, ik, "duplicated fp_forces values differ from full_fp_neb's own "
                                               "record by more than 5e-4 eV/A"))
                diag_energy = img.get("fp_energy_total_eV")
                full_energy = full_img.get("fp_energy_total_eV")
                # Tight numerical tolerance, not exact equality: real FPBench
                # data round-trips through JSON with up to ~1.5e-5 eV of
                # float32-vs-float64 serialization noise between the two
                # branches' independently-stored copies of the same value
                # (confirmed by direct inspection, exact-power-of-2-sized
                # diffs -- a quantization artifact, not a real discrepancy).
                # A genuine data inconsistency would be orders of magnitude
                # larger than this.
                if (diag_energy is not None and full_energy is not None
                        and not np.isclose(diag_energy, full_energy, rtol=0, atol=5e-4)):
                    dft_on_fp_structure_mismatches.append(
                        (fp_key, pkey, ik, "duplicated fp_energy_total_eV differs from full_fp_neb's own "
                                           f"record by more than 5e-4 eV: {diag_energy} vs {full_energy}"))
    record("dft_static_on_fp_neb structures/species-order/atom-count correspond exactly to the same "
           "FP's full_fp_neb image at the same index; duplicated fp energy/forces (where both branches "
           "store them) match exactly",
           len(dft_on_fp_structure_mismatches) == 0, dft_on_fp_structure_mismatches[:5])

    # -- dft_static_on_fp_neb <-> full_fp_neb image-count correspondence --
    dft_on_fp_mismatches = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        full = model.get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {})
        for pkey, pdata in model.get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("pathways", {}).items():
            full_pdata = full.get(pkey)
            if full_pdata is None:
                continue
            n_diag = len(pdata.get("images", {}))
            n_full = len(full_pdata.get("final_fp_neb_images", {}))
            if n_diag > n_full:
                dft_on_fp_mismatches.append((fp_key, pkey, n_diag, n_full))
    record("dft_static_on_fp_neb image counts do not exceed full_fp_neb's (where both present)",
           len(dft_on_fp_mismatches) == 0, dft_on_fp_mismatches[:5])

    # -- duplicate pathway records within one protocol branch --
    dup_pathway_records = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB, PROTOCOL_DFT_STATIC_ON_FP_NEB):
            keys = list(model.get(protocol, {}).get("pathways", {}).keys())
            if len(keys) != len(set(keys)):
                dup_pathway_records.append((fp_key, protocol))
    record("no duplicate pathway keys within any protocol branch",
           len(dup_pathway_records) == 0, dup_pathway_records[:5])

    # -- status / convergence, explicit non-converged records preserved --
    missing_status = []
    for fp_key in fp_order:
        for pkey, pdata in models.get(fp_key, {}).get(PROTOCOL_FULL_FP_NEB, {}).get("pathways", {}).items():
            if "neb_status" not in pdata or "neb_converged" not in pdata.get("neb_status", {}):
                missing_status.append((fp_key, pkey))
    record("full_fp_neb pathways carry an explicit neb_status.neb_converged value (never defaulted)",
           len(missing_status) == 0, missing_status[:5])

    # -- unsuccessful-record branches (full_fp_neb/fp_static_on_dft_neb
    #    "unsuccessful_pathways", dft_static_on_fp_neb
    #    "unsuccessful_image_attempts"): identifiers and status values only.
    #    Nothing here feeds any scientific metric -- see
    #    collect_unsuccessful_records's docstring. A record co-present in
    #    both the unsuccessful branch and the active "pathways"/"images"
    #    branch for the same key is flagged as ambiguous, never silently
    #    resolved in either direction. --
    unsuccessful_issues = []
    for fp_key in fp_order:
        model = models.get(fp_key, {})
        for protocol in (PROTOCOL_FULL_FP_NEB, PROTOCOL_FP_STATIC_ON_DFT_NEB):
            active_pathways = model.get(protocol, {}).get("pathways", {})
            for pkey, urec in model.get(protocol, {}).get("unsuccessful_pathways", {}).items():
                ident = urec.get("identifiers", {})
                if f"{ident.get('icsd_id')}|{ident.get('source_path_id')}" != pkey:
                    unsuccessful_issues.append((fp_key, protocol, pkey, "identifiers do not match key"))
                status = urec.get("calculation_status")
                if status not in ("not_run", "missing", "interrupted", "failed"):
                    unsuccessful_issues.append((fp_key, protocol, pkey, f"unexpected calculation_status {status!r}"))
                if pkey in active_pathways:
                    unsuccessful_issues.append((fp_key, protocol, pkey,
                                                 "also present in 'pathways' (ambiguous record)"))
        for attempt_key, urec in model.get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("unsuccessful_image_attempts", {}).items():
            expected_key = f"{urec.get('fp_key')}|{urec.get('pathway_key')}|{urec.get('image_index')}"
            if expected_key != attempt_key:
                unsuccessful_issues.append((fp_key, PROTOCOL_DFT_STATIC_ON_FP_NEB, attempt_key,
                                             "identifiers do not match key"))
            status = urec.get("status")
            if status not in ("not_run", "missing", "interrupted", "failed", "rejected"):
                unsuccessful_issues.append((fp_key, PROTOCOL_DFT_STATIC_ON_FP_NEB, attempt_key,
                                             f"unexpected status {status!r}"))
    record("unsuccessful-record branches (unsuccessful_pathways / unsuccessful_image_attempts) have "
           "consistent identifiers and a valid status value, and never duplicate an active pathway/image "
           "record for the same key",
           len(unsuccessful_issues) == 0, unsuccessful_issues[:5])

    summary_df = pd.DataFrame(checks)
    ok = bool(summary_df["ok"].all()) if len(summary_df) else False
    return type("ValidationReport", (), {"summary_df": summary_df, "ok": ok, "checks": checks})()


class NEBAnalysisResults:
    """Canonical NEB analysis objects returned by build_neb_analysis_results(...).
    Every attribute is one of the existing, already-verified canonical
    objects/tables produced by the functions in this module -- this class
    only orchestrates and names them, it computes nothing itself.

    Attributes:
      fp_order                                list of FP keys analyzed.
      validation_report                       return of validate_neb_analysis_inputs(...),
                                               or None if validate=False.
      protocol_coverage_df                    per-FP, per-protocol coverage table
                                               (build_protocol_coverage_table).
      material_metadata_df                    one row per unique icsd_id (build_material_metadata).
      dft_neb_images_by_path                  {pathway_key: [ASE Atoms, ...]}.
      full_fp_neb_images_by_fp_path           {fp_key: {pathway_key: [ASE Atoms, ...]}}.
      fp_static_on_dft_neb_images_by_fp_path  {fp_key: {pathway_key: [ASE Atoms, ...]}}.
      full_fp_neb_status_by_fp_path           {fp_key: {(icsd_str, path_str): status_record}}.
      dft_path_metrics_df                     DFT pathway metrics and classification.
      dft_valid_path_metrics_df               dft_path_metrics_df with Invalid pathways excluded.
      fp_path_metrics_by_protocol             {fp_key: {protocol: DataFrame}} FP pathway
                                               metrics/classification, by protocol.
      endpoint_rmsd_by_fp_protocol_path       {fp_key: {protocol: {pathway_key: rmsd_record}}}.
      full_fp_neb_path_force_errors_df        force errors on the FP-NEB path (protocol
                                               dft_static_on_fp_neb). None if that protocol has
                                               no data for any FP -- never fabricated.
      dft_neb_path_force_errors_df            force errors on the DFT-NEB path (protocol
                                               fp_static_on_dft_neb). None if that protocol has
                                               no data for any FP -- never fabricated.
      coverage_mismatch_df                    output of validate_analysis_coverage(...);
                                               empty DataFrame if no mismatches found.
      unsuccessful_records_by_fp_protocol     {fp_key: {protocol: {record_key: record}}},
                                               output of collect_unsuccessful_records(...):
                                               failed/missing/interrupted/not_run full_fp_neb
                                               and fp_static_on_dft_neb pathways, and rejected/
                                               failed/missing/not_run dft_static_on_fp_neb image
                                               attempts. Status/coverage reporting only -- never
                                               read by any table above, never affects a
                                               denominator (see that function's docstring).
      barrier_error_summary_by_protocol       {protocol: DataFrame} full FP-NEB and
                                               static FP-on-DFT-NEB barrier/energy-range results.
      full_fp_neb_convergence_summary         {fp_key: {protocol: {n_total, n_neb_not_conv}}};
                                               denominators/eligibility counts for the barrier
                                               summaries (n_neb_not_conv is the string
                                               "not applicable" for fp_static_on_dft_neb).
      profile_summary_by_protocol             {protocol: DataFrame} energy-profile shape
                                               agreement, endpoint ranking/energy-difference
                                               error, and integrated-area results.
      path_records_full                       per-path full_fp_neb records (area, fwd_err,
                                               bwd_err, topo_correct, rank_correct,
                                               neb_converged) used by non-converged diagnostics.

    Failed / missing / non-converged record semantics: this module and the
    real FPBench data it operates on distinguish exactly three states, and
    deliberately do not carry a richer status enum (e.g. a
    status: completed|failed|missing|not_run field with failure_stage/
    error_type/error_message) because no calculation in the real 154-pathway
    dataset has ever genuinely errored/crashed -- only "ran and converged" and
    "ran and did not converge" occur in practice, so a failure taxonomy with
    no real data to populate or verify it would be speculative machinery, not
    a documented fact (the project's standing rule against building for
    hypothetical requirements: see CLAUDE.md).
      1. Present and converged      -- neb_status.neb_converged == True.
      2. Present and non-converged  -- neb_status.neb_converged == False.
         Preserved verbatim, never dropped and never silently treated as
         converged (see test 7 in test_neb_analysis.py).
      3. Missing/unavailable        -- a pathway key absent from
         reference_data["pathways"] despite being listed in
         common_pathway_keys (see test 22), or an entire protocol branch
         absent/empty for a given FP (see test 23). Both cases surface as
         None or an empty table on the relevant NEBAnalysisResults attribute
         -- never fabricated, never silently substituted from another
         protocol or FP.
    If a genuinely failed (errored, not merely non-converged) calculation is
    ever produced, add it as a fourth documented case here, backed by a real
    example, rather than pre-building an unused schema for it now.
    """

    def __init__(self, **kwargs):
        self._fields = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        lines = ["NEBAnalysisResults("]
        for k in self._fields:
            v = getattr(self, k)
            if isinstance(v, pd.DataFrame):
                desc = f"DataFrame[{len(v)} rows]" if v is not None else "None"
            elif isinstance(v, dict):
                desc = f"dict[{len(v)} keys]"
            elif isinstance(v, list):
                desc = f"list[{len(v)}]"
            else:
                desc = repr(v) if v is None or isinstance(v, (bool, int, float, str)) else type(v).__name__
            lines.append(f"  {k} = {desc}")
        lines.append(")")
        return "\n".join(lines)


def build_neb_analysis_results(reference_data, fp_results, expected_pathways=None,
                                fp_order=None, outlier_threshold=1.0, validate=True):
    """Public entry point: build every canonical NEB analysis object from a
    DFT-NEB reference and FP results, both following the canonical JSON
    schema documented in input_data/README.md, results/README.md, and,
    briefly, the analysis notebook's "Using FPBench data or another NEB
    dataset" section. This is the one function external users should call;
    every object it returns comes from the same, already-verified functions
    this notebook itself uses (no new scientific definitions, thresholds, or
    formulas are introduced here).

    Parameters
    ----------
    reference_data : dict or str or pathlib.Path
        The DFT-NEB reference, already loaded as a dict (matching
        input_data/ion_migration_neb_reference.json's schema), or a path
        to such a JSON file (loaded with load_neb_datasets-equivalent
        json.load; file paths are supported for convenience, dicts are the
        primary supported input).
    fp_results : dict or str or pathlib.Path
        The FP results, already loaded as a dict (matching
        results/ion_migration_neb_results.json's schema: a top-level
        "models" key mapping fp_key -> {protocol -> {"pathways": {...}}}),
        or a path to such a JSON file.
    expected_pathways : int or None, default None
        If given, checked as an informational count during validation (see
        validate_neb_analysis_inputs). The FPBench benchmark passes 154
        here; external datasets should leave this None -- no other
        behavior in this function assumes any particular pathway count.
    fp_order : list of str or None, default None
        FP keys to analyze, in display order. Defaults to
        list(fp_results["models"].keys()) if not given.
    outlier_threshold : float, default 1.0
        eV threshold used by the barrier-error summaries' threshold-
        filtered MAE/RMSE columns (same meaning as the notebook's
        OUTLIER_THRESHOLD). Does not affect any other metric.
    validate : bool, default True
        If True, runs validate_neb_analysis_inputs(...) first and stores
        the report on the returned object as .validation_report (does not
        raise on a failed check -- inspect .validation_report.ok /
        .summary_df). If False, .validation_report is None and no checks
        run before building the analysis objects (data is still preserved
        as-is either way; validation never filters anything).

    Returns
    -------
    NEBAnalysisResults
        See that class's docstring for the full attribute list. Any
        protocol with no data for any FP is represented explicitly (empty
        DataFrame / None for the two force-error tables) rather than
        fabricated or filled in from another protocol -- e.g. if no FP
        supplies dft_static_on_fp_neb results, full_fp_neb_path_force_errors_df
        is None, and the corresponding "Force errors on the FP-NEB path"
        analysis is unavailable, not silently substituted with data from
        fp_static_on_dft_neb or any other protocol.

    Notes
    -----
    Supplying only a subset of the three protocols is supported: pass an
    empty models[fp_key] for any protocol you do not have (or omit the
    fp_key entirely for a protocol's sub-dict) and the corresponding
    canonical objects come back empty/None rather than raising, so the
    parts of the analysis that only depend on the protocols you do have
    still work. Missing, failed, and non-converged pathway records must
    still be present in the input (with their real status/convergence
    fields) for the metric eligibility logic (e.g. the NEB-converged
    filter in compute_barrier_error_summaries) to work correctly; do not
    remove them before calling this function.
    """
    if isinstance(reference_data, (str, Path)):
        with open(reference_data) as f:
            reference_data = json.load(f)
    if isinstance(fp_results, (str, Path)):
        with open(fp_results) as f:
            fp_results = json.load(f)

    # Accept either the canonical nested DFT-reference schema
    # (pathways[key]["dft_neb_reference"]["images"]) or the legacy flat
    # schema (pathways[key]["dft_neb_images"]) that every function below
    # this point was originally written against. Raises ValueError if both
    # are supplied and disagree -- never silently resolved.
    reference_data = _normalize_reference_data(reference_data)

    if fp_order is None:
        fp_order = list(fp_results.get("models", {}).keys())

    validation_report = (
        validate_neb_analysis_inputs(reference_data, fp_results,
                                      expected_pathways=expected_pathways, fp_order=fp_order)
        if validate else None
    )

    protocol_coverage_df = build_protocol_coverage_table(reference_data, fp_results, fp_order)
    material_metadata_df = build_material_metadata(reference_data)

    dft_neb_images_by_path, dft_pathway_keys = build_dft_neb_image_map(reference_data)
    full_fp_neb_images_by_fp_path = build_full_fp_neb_image_map(fp_results, fp_order)
    fp_static_on_dft_neb_images_by_fp_path = build_fp_static_on_dft_neb_image_map(
        fp_results, reference_data, fp_order)
    full_fp_neb_status_by_fp_path = build_full_fp_neb_status_map(fp_results, fp_order)

    dft_path_metrics_df = build_dft_path_metrics(dft_neb_images_by_path, dft_pathway_keys)
    fp_path_metrics_by_protocol = build_fp_path_metrics(
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path, fp_order)

    dft_valid_path_metrics_df = dft_path_metrics_df[
        dft_path_metrics_df["Pathway Topology"] != "Invalid"].copy()
    if not dft_valid_path_metrics_df.empty:
        dft_valid_path_metrics_df["ICSD"] = dft_valid_path_metrics_df["ICSD"].astype(int)
        dft_valid_path_metrics_df["Path"] = dft_valid_path_metrics_df["Path"].astype(int)

    endpoint_rmsd_by_fp_protocol_path = compute_endpoint_rmsd(
        fp_path_metrics_by_protocol, dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path, fp_order)

    coverage_mismatch_df = validate_analysis_coverage(
        fp_static_on_dft_neb_images_by_fp_path, dft_neb_images_by_path, fp_order)

    unsuccessful_records_by_fp_protocol = collect_unsuccessful_records(fp_results, fp_order)

    # Force-error tables: build only the ones with actual data present for
    # at least one FP; never fabricate or substitute from another protocol.
    dft_static_on_fp_neb_records_by_fp = {
        fp_key: fp_results.get("models", {}).get(fp_key, {})
                          .get(PROTOCOL_DFT_STATIC_ON_FP_NEB, {}).get("pathways", {})
        for fp_key in fp_order
    }
    full_fp_neb_path_force_errors_df = (
        build_full_fp_neb_path_force_errors(dft_static_on_fp_neb_records_by_fp, fp_order)
        if any(dft_static_on_fp_neb_records_by_fp.values()) else None
    )
    dft_neb_path_force_errors_df = (
        build_dft_neb_path_force_errors(fp_static_on_dft_neb_images_by_fp_path, dft_neb_images_by_path, fp_order)
        if any(fp_static_on_dft_neb_images_by_fp_path.values()) else None
    )

    # benchmark_pathways_df: for FPBench this is the 154 common_pathway_keys;
    # for an external dataset with no such field, every DFT-reference
    # pathway is in scope (i.e. this join becomes a no-op, not a filter).
    pathway_keys = reference_data.get("common_pathway_keys") or list(reference_data.get("pathways", {}).keys())
    benchmark_pathways_df = pd.DataFrame(
        [{"ICSD": int(icsd), "Path": int(path)} for icsd, path in (k.split("|") for k in pathway_keys)]
    ) if pathway_keys else pd.DataFrame(columns=["ICSD", "Path"])

    barrier_error_summary_by_protocol, full_fp_neb_convergence_summary = compute_barrier_error_summaries(
        dft_valid_path_metrics_df, fp_path_metrics_by_protocol, full_fp_neb_status_by_fp_path,
        endpoint_rmsd_by_fp_protocol_path, benchmark_pathways_df, fp_order, outlier_threshold,
    )

    # area_between_curves/simplify_class live in neb_plots.py (plotting-
    # adjacent numeric helpers); imported here, lazily, only inside this
    # public entry point, so this module's own top-level import surface
    # stays plotting-free (matplotlib etc. are not imported at module load).
    from neb_plots import area_between_curves, simplify_class

    profile_summary_by_protocol, path_records_full = compute_profile_summaries(
        fp_path_metrics_by_protocol, dft_path_metrics_df, dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path, fp_static_on_dft_neb_images_by_fp_path,
        full_fp_neb_status_by_fp_path, fp_order, area_between_curves, simplify_class,
    )

    return NEBAnalysisResults(
        fp_order=fp_order,
        validation_report=validation_report,
        protocol_coverage_df=protocol_coverage_df,
        material_metadata_df=material_metadata_df,
        dft_neb_images_by_path=dft_neb_images_by_path,
        full_fp_neb_images_by_fp_path=full_fp_neb_images_by_fp_path,
        fp_static_on_dft_neb_images_by_fp_path=fp_static_on_dft_neb_images_by_fp_path,
        full_fp_neb_status_by_fp_path=full_fp_neb_status_by_fp_path,
        dft_path_metrics_df=dft_path_metrics_df,
        dft_valid_path_metrics_df=dft_valid_path_metrics_df,
        fp_path_metrics_by_protocol=fp_path_metrics_by_protocol,
        endpoint_rmsd_by_fp_protocol_path=endpoint_rmsd_by_fp_protocol_path,
        full_fp_neb_path_force_errors_df=full_fp_neb_path_force_errors_df,
        dft_neb_path_force_errors_df=dft_neb_path_force_errors_df,
        coverage_mismatch_df=coverage_mismatch_df,
        unsuccessful_records_by_fp_protocol=unsuccessful_records_by_fp_protocol,
        barrier_error_summary_by_protocol=barrier_error_summary_by_protocol,
        full_fp_neb_convergence_summary=full_fp_neb_convergence_summary,
        profile_summary_by_protocol=profile_summary_by_protocol,
        path_records_full=path_records_full,
    )


def example_canonical_pathway_records():
    """Small, documented, runnable example of the canonical record schema
    for all three protocols plus an explicit failed/non-converged record --
    NOT a generic converter for arbitrary external formats (this project
    does not attempt to guess unknown schemas; adapt this template by hand
    to your own data instead). Every structure below is a real pymatgen
    Structure.as_dict() output (a 2-atom NaCl-like cell), so the schema is
    authentic, not invented -- this is the same "structure" field shape
    used throughout input_data/ion_migration_neb_reference.json and
    results/ion_migration_neb_results.json.

    Returns a dict with five keys, each adaptable independently:
      "dft_reference_pathway"        one DFT-NEB reference pathway (3 images:
                                      initial, one intermediate, final).
      "full_fp_neb_pathway"          one full FP-NEB result for that pathway
                                      (converged).
      "fp_static_on_dft_neb_pathway" one static FP evaluation on that
                                      pathway's DFT-NEB images (note: no
                                      "structure" field per image -- this
                                      protocol reuses the DFT-NEB reference
                                      structures at the same image index).
      "dft_static_on_fp_neb_pathway" one DFT-static-on-FP-NEB diagnostic
                                      result for that pathway.
      "failed_full_fp_neb_pathway"   a second pathway's full FP-NEB result
                                      that did NOT converge, showing how a
                                      non-converged record is represented
                                      (neb_status.neb_converged: False) --
                                      preserve records like this as-is; do
                                      not drop them before calling
                                      build_neb_analysis_results(...)."""
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(4.0)
    struct_initial = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    struct_mid = Structure(lattice, ["Na", "Cl"], [[0.15, 0.0, 0.0], [0.5, 0.5, 0.5]])
    struct_final = Structure(lattice, ["Na", "Cl"], [[0.3, 0.0, 0.0], [0.5, 0.5, 0.5]])

    def _img(index, role, struct, energy, forces):
        return {"image_index": index, "endpoint_role": role, "structure": struct.as_dict(),
                "energy_total_eV": energy, "forces_eV_per_angstrom": forces}

    # Small but nonzero everywhere (relaxed endpoints have small residual
    # forces in practice too): keeps every force-angle computation in this
    # example defined, so running it never triggers a RuntimeWarning (see
    # compute_force_angle_error / _safe_nanmean / _safe_nanmax). A genuinely
    # all-zero-force image is exercised separately, deliberately, in
    # test_neb_analysis.py to prove the zero-valid-angle path itself stays
    # warning-free -- it is not used here, where a runnable, warning-free
    # example is the goal.
    zero_forces = [[0.002, 0.001, -0.001], [-0.002, -0.001, 0.001]]
    mid_forces = [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]

    dft_reference_pathway = {
        "identifiers": {"icsd_id": "999001", "source_path_id": "1",
                         "migrating_species": "Na", "sum_formula": "Na1 Cl1",
                         "is_common_path": True},
        "status": "present",
        "initial_source_endpoint": None,
        "final_source_endpoint": None,
        "dft_relaxed_endpoints": {"initial": _img(0, "initial", struct_initial, -10.000, zero_forces),
                                   "final": _img(2, "final", struct_final, -10.000, zero_forces)},
        "dft_neb_images": [
            _img(0, "initial", struct_initial, -10.000, zero_forces),
            _img(1, "intermediate", struct_mid, -9.700, mid_forces),
            _img(2, "final", struct_final, -10.000, zero_forces),
        ],
    }

    full_fp_neb_pathway = {
        "identifiers": {"icsd_id": "999001", "source_path_id": "1"},
        "endpoint_relaxation": {"note": "example record"},
        "neb_status": {"neb_converged": True, "neb_last_fmax": 0.03, "neb_n_steps": 20,
                        "status_source": "example"},
        "convergence": None,
        "steps": 20,
        "barrier_metadata_verbatim": {"forward_barrier_eV": 0.31, "backward_barrier_eV": 0.31,
                                       "delta_E": 0.0, "energy_range": 0.31},
        "final_fp_neb_images": {
            "0": {"structure": struct_initial.as_dict(), "fp_energy_total_eV": -9.98,
                  "fp_forces_eV_per_angstrom": zero_forces},
            "1": {"structure": struct_mid.as_dict(), "fp_energy_total_eV": -9.68,
                  "fp_forces_eV_per_angstrom": mid_forces},
            "2": {"structure": struct_final.as_dict(), "fp_energy_total_eV": -9.98,
                  "fp_forces_eV_per_angstrom": zero_forces},
        },
    }

    fp_static_on_dft_neb_pathway = {
        "identifiers": {"icsd_id": "999001", "source_path_id": "1"},
        "status": "present",
        # No "structure" key per image: this protocol single-point-evaluates
        # the FP on the DFT-NEB reference's own structures at the same
        # image index -- it has no structure of its own to store.
        "images": {
            "0": {"fp_energy_total_eV": -9.97, "fp_forces_eV_per_angstrom": zero_forces},
            "1": {"fp_energy_total_eV": -9.65, "fp_forces_eV_per_angstrom": mid_forces},
            "2": {"fp_energy_total_eV": -9.97, "fp_forces_eV_per_angstrom": zero_forces},
        },
    }

    dft_static_on_fp_neb_pathway = {
        "identifiers": {"icsd_id": "999001", "source_path_id": "1"},
        "status": "present (example population)",
        "source_population": "example",
        "images": {
            "0": {"dft_energy_total_eV": -10.000, "dft_forces_eV_per_angstrom": zero_forces,
                  "fp_structure": struct_initial.as_dict(), "fp_energy_total_eV": -9.98,
                  "fp_forces_eV_per_angstrom": zero_forces},
            "1": {"dft_energy_total_eV": -9.700, "dft_forces_eV_per_angstrom": mid_forces,
                  "fp_structure": struct_mid.as_dict(), "fp_energy_total_eV": -9.68,
                  "fp_forces_eV_per_angstrom": mid_forces},
            "2": {"dft_energy_total_eV": -10.000, "dft_forces_eV_per_angstrom": zero_forces,
                  "fp_structure": struct_final.as_dict(), "fp_energy_total_eV": -9.98,
                  "fp_forces_eV_per_angstrom": zero_forces},
        },
    }

    failed_full_fp_neb_pathway = {
        "identifiers": {"icsd_id": "999002", "source_path_id": "1"},
        "endpoint_relaxation": {"note": "example record"},
        # Explicit non-convergence, preserved as-is -- never defaulted to
        # True and never dropped from the input.
        "neb_status": {"neb_converged": False, "neb_last_fmax": 0.42, "neb_n_steps": 1000,
                        "status_source": "example"},
        "convergence": None,
        "steps": 1000,
        "barrier_metadata_verbatim": {"forward_barrier_eV": 0.55, "backward_barrier_eV": 0.55,
                                       "delta_E": 0.0, "energy_range": 0.55},
        "final_fp_neb_images": {
            "0": {"structure": struct_initial.as_dict(), "fp_energy_total_eV": -9.9,
                  "fp_forces_eV_per_angstrom": zero_forces},
            "1": {"structure": struct_mid.as_dict(), "fp_energy_total_eV": -9.4,
                  "fp_forces_eV_per_angstrom": mid_forces},
            "2": {"structure": struct_final.as_dict(), "fp_energy_total_eV": -9.9,
                  "fp_forces_eV_per_angstrom": zero_forces},
        },
    }

    return {
        "dft_reference_pathway": dft_reference_pathway,
        "full_fp_neb_pathway": full_fp_neb_pathway,
        "fp_static_on_dft_neb_pathway": fp_static_on_dft_neb_pathway,
        "dft_static_on_fp_neb_pathway": dft_static_on_fp_neb_pathway,
        "failed_full_fp_neb_pathway": failed_full_fp_neb_pathway,
    }
