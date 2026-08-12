"""
FPBench standard force-results schema.

This module defines the one standard per-model force-results format used
throughout FPBench, and a small public entry point — `build_force_results` —
that lets a user build that same format from their own Cartesian DFT/FP
forces, so their dataset can be passed directly to the FPBench analysis
functions.

Standard per-model schema
--------------------------
    force_results[model] = {
        "dft_force_magnitude":   list[float]  len = n_atoms   |F_DFT|  (eV/Å)
        "fp_force_magnitude":    list[float]  len = n_atoms   |F_FP|   (eV/Å)
        "force_magnitude_error": list[float]  len = n_atoms   Δ|F| = |F_FP| - |F_DFT|  (signed, eV/Å)
        "force_angle_error":     list[float]  len = n_atoms   Δθ (degrees; NaN where undefined)
        "force_vector_error":    list[float]  len = n_atoms   e_vec = ||F_FP - F_DFT||  (eV/Å)

        # optional provenance — present only when available and validated
        "structure_id": list[...]  len = n_atoms
        "atom_index":   list[int]  len = n_atoms
    }

`force_vector_error` (e_vec) is a Euclidean vector-difference norm — it is
NOT a force-magnitude error and must never be treated as one. Use
`np.abs(force_magnitude_error)` when |Δ|F|| is needed; there is no separate
stored field for it.

File-level wrapper, when a standardized dataset is written to JSON:
    {
        "schema_version": "1.0",
        "dataset_name": "...",
        "units": {"force": "eV/Å", "angle": "degree"},
        "models": {model: {...per-model schema above...}, ...}
    }

After loading such a file, analyses operate on `data["models"]`.
"""

import numpy as np

SCHEMA_VERSION = "1.0"

# Numerical floor distinguishing a genuinely zero force vector (angle
# undefined) from real physics. This is NOT the manuscript's FDFT_MIN
# (0.01 eV/Å) atom-selection cutoff — it only guards against division by
# zero / arccos of a degenerate ratio.
_ZERO_FORCE_TOL = 1e-12


def _atom_count(structure_forces, label):
    arr = np.asarray(structure_forces, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{label} must be an (n_atoms, 3) Cartesian force array, "
            f"got shape {arr.shape}"
        )
    return arr


def build_force_results(dft_forces, fp_forces, structure_ids=None):
    """
    Build the standard FPBench force_results dict from Cartesian forces.

    Parameters
    ----------
    dft_forces : sequence of array-like, shape (N_i, 3)
        One Cartesian DFT force array per structure, in eV/Å.
    fp_forces : dict[str, sequence of array-like, shape (N_i, 3)]
        One entry per FP model; each value is a per-structure list of
        Cartesian FP force arrays, same length and atom ordering as
        `dft_forces`.
    structure_ids : sequence, optional
        One identifier per structure, same length as `dft_forces`. When
        given, the returned per-model dicts carry `structure_id`/`atom_index`
        provenance. When omitted, those fields are left out entirely —
        core analyses do not require them.

    Returns
    -------
    dict[str, dict]
        force_results[model] following the standard schema documented in
        this module's docstring.

    Raises
    ------
    ValueError
        If DFT/FP structure counts, per-structure atom counts, or
        `structure_ids` length do not match. Mismatches are never silently
        truncated.
    """
    n_structures = len(dft_forces)
    if structure_ids is not None and len(structure_ids) != n_structures:
        raise ValueError(
            f"structure_ids has {len(structure_ids)} entries but "
            f"dft_forces has {n_structures} structures"
        )

    dft_checked = [
        _atom_count(f, f"dft_forces[{i}]") for i, f in enumerate(dft_forces)
    ]
    dft_natoms = [f.shape[0] for f in dft_checked]

    force_results = {}
    for model, model_forces in fp_forces.items():
        if len(model_forces) != n_structures:
            raise ValueError(
                f"fp_forces[{model!r}] has {len(model_forces)} structures "
                f"but dft_forces has {n_structures} structures"
            )

        dF_parts, angle_parts, evec_parts = [], [], []
        Fdft_parts, Ffp_parts = [], []
        sid_parts, aidx_parts = [], []

        for i in range(n_structures):
            F_dft = dft_checked[i]
            F_fp = _atom_count(model_forces[i], f"fp_forces[{model!r}][{i}]")

            if F_fp.shape[0] != dft_natoms[i]:
                raise ValueError(
                    f"Structure {i}: dft_forces has {dft_natoms[i]} atoms "
                    f"but fp_forces[{model!r}][{i}] has {F_fp.shape[0]} atoms. "
                    "DFT and FP forces must have matching atom counts and "
                    "ordering per structure."
                )

            mag_dft = np.linalg.norm(F_dft, axis=-1)
            mag_fp = np.linalg.norm(F_fp, axis=-1)

            dF = mag_fp - mag_dft
            e_vec = np.linalg.norm(F_fp - F_dft, axis=-1)

            denom = mag_dft * mag_fp
            safe_denom = np.where(denom > _ZERO_FORCE_TOL, denom, 1.0)
            cosine = np.einsum("ij,ij->i", F_dft, F_fp) / safe_denom
            cosine = np.clip(cosine, -1.0, 1.0)
            theta = np.degrees(np.arccos(cosine))
            theta = np.where(denom > _ZERO_FORCE_TOL, theta, np.nan)

            Fdft_parts.append(mag_dft)
            Ffp_parts.append(mag_fp)
            dF_parts.append(dF)
            angle_parts.append(theta)
            evec_parts.append(e_vec)

            if structure_ids is not None:
                sid_parts.append(np.full(dft_natoms[i], structure_ids[i], dtype=object))
                aidx_parts.append(np.arange(dft_natoms[i]))

        entry = {
            "dft_force_magnitude": np.concatenate(Fdft_parts).tolist(),
            "fp_force_magnitude": np.concatenate(Ffp_parts).tolist(),
            "force_magnitude_error": np.concatenate(dF_parts).tolist(),
            "force_angle_error": np.concatenate(angle_parts).tolist(),
            "force_vector_error": np.concatenate(evec_parts).tolist(),
        }
        if structure_ids is not None:
            entry["structure_id"] = np.concatenate(sid_parts).tolist()
            entry["atom_index"] = np.concatenate(aidx_parts).tolist()

        force_results[model] = entry

    return force_results
