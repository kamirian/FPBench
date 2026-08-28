# Adding a potential

This document describes how to run an FPBench benchmark on a foundation potential (FP) that is
not already reported on the leaderboard.

## Requirements

- An **ASE-compatible calculator** for the FP. A few lines of Python ending in a variable named
  `calc` that exposes `get_potential_energy()` and `get_forces()` are sufficient.
- A **Python environment** in which that calculator imports and runs. FP packages frequently pin
  conflicting PyTorch and ASE versions, so a separate virtual environment per FP family is
  recommended rather than a single shared environment.
- A **checkpoint file**, where the FP requires one. Some packages, CHGNet for example, load their
  own default weights and need no path.
- Access to a machine or cluster capable of running the calculations. FPBench does not execute
  them and does not install packages.

No metric or analysis code requires modification. Registering a potential is the only edit
needed.

---

## Registry structure

Each component contains a generator notebook holding a `POTENTIAL_REGISTRY` dictionary. Every
entry describes one FP: the environment it runs in, the checkpoint it loads, and the code that
constructs its ASE calculator. The generator expands that entry into standalone run scripts, one
per job, which are then submitted on the target cluster.

Adding an FP consists of adding one dictionary entry.

| Component | Generator notebook |
|---|---|
| Force prediction | `Force_error/generation/matpes_PBE_run_generator.ipynb`, and the `r2scan` and `omat24` variants |
| Phase stability and elemental ordering | `Phase_stability_ordering/generation/convexhull_ordering_run_generator.ipynb` |
| Ion migration by NEB | `Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb` |

---

## Registry fields

Each component's registry records the same four quantities: the environment to run in, the
checkpoint to load, the code that constructs the ASE calculator, and the name written into the
results. The field names differ between components and are documented alongside the registry to
which they belong.

- [Force prediction, Registering a potential](Force_error/README.md#registering-a-potential)
- [Phase stability and ordering, Registering a potential](Phase_stability_ordering/README.md#registering-a-potential)
- [Ion migration by NEB, Registering a potential](Ion_migration_NEB/README.md#registering-a-potential)

In every component the calculator code refers to the checkpoint through the variable
`MODEL_PATH`, which the generator substitutes from the entry's `model_path`. An existing entry in
the same notebook should be used as the template, rather than an entry from another component.

---

## Example

The simplest case is an FP whose package supplies its own weights, for which `model_path` is
`None`. The following is the CHGNet entry from the phase stability generator.

```python
"chgnet": {
    "mlip_name":     "chgnet",
    "site_pkgs":     "/path/to/chgnet_env/lib/python3.10/site-packages",
    "venv_activate": "/path/to/chgnet_env/bin/activate",
    "python":        "/path/to/chgnet_env/bin/python",
    "model_path":    None,
    "calc_setup":    textwrap.dedent("""
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator()
    """).strip(),
},
```

An FP that loads a checkpoint refers to it as `MODEL_PATH` in its setup code.

```python
"my_fp": {
    "mlip_name":     "my_fp",
    "site_pkgs":     "/path/to/my_fp_env/lib/python3.10/site-packages",
    "venv_activate": "/path/to/my_fp_env/bin/activate",
    "python":        "/path/to/my_fp_env/bin/python",
    "model_path":    "/path/to/checkpoints/my_fp_model.pt",
    "calc_setup":    textwrap.dedent("""
        from my_fp.calculators import MyFPCalculator
        calc = MyFPCalculator(
            model_paths=[MODEL_PATH],
            device="cpu",
            default_dtype="float64",
        )
    """).strip(),
},
```

The placeholder paths are checked. Values still beginning with `/path/to/` or `YOUR_` raise an
explicit error rather than producing a job script that fails later on the cluster.

---

## Procedure

1. **Register the potential** as described above.
2. **Configure the notebook.** Set the active potential or FP key list in the configuration
   section, together with the dataset, output, and SLURM settings appropriate to the target
   system.
3. **Generate.** Running the generation cells writes run scripts and submission scripts into the
   output directory. No job is submitted at this stage.
4. **Submit** the generated scripts on the cluster.
5. **Merge.** Once the jobs complete, the merge section validates each job's output against the
   shared reference and writes a standardized results file.
6. **Analyse.** The component's analysis notebook reads the merged file and produces the same
   metric tables reported on the leaderboard.

Steps 3 to 6 are documented per component under "Using FPBench" and "Quick start" in the
corresponding README.

---

## Notes

- **One environment per FP family.** FP packages pin incompatible PyTorch and ASE versions, and
  each registry entry records its own environment so that they need not be shared.
- **No automatic installation.** Where a package is missing, the generated script raises an
  explicit error rather than invoking `pip install`.
- **Generation is separate from submission.** Running the generator cells does not launch a job.
- **Existing results are preserved.** Registry keys and result names are asserted unique, so a
  collision fails explicitly rather than overwriting another FP's results.
- **Results are tied to a model version.** The exact checkpoint should be recorded when a
  benchmark is reported. A later version of the same architecture constitutes a separate entry
  rather than an update to an existing one.

---

## Application to an independent dataset

Where DFT and FP results already exist and only the metrics are required, the generators are not
needed. Each component's analysis functions are importable from its `scripts/` directory and
accept standardized inputs directly. See the "Using FPBench" and "Required inputs and outputs"
sections of the component README.
