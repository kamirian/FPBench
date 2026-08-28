# Adding a potential

How to run an FPBench benchmark on a foundation potential (FP) that is not already on the
leaderboard.

## What you need

- An **ASE-compatible calculator** for your FP. If you can write a few lines of Python that end
  with a variable named `calc` exposing `get_potential_energy()` and `get_forces()`, you have
  everything FPBench needs.
- A **Python environment** where that calculator imports and runs. FP packages often pin
  conflicting PyTorch and ASE versions, so we strongly recommend one virtual environment per FP
  family rather than a single shared one.
- A **checkpoint file**, if your FP loads one. Some packages (CHGNet, for example) ship their own
  default weights and need no path.
- Access to a machine or cluster that can run the calculations. FPBench does not run them for
  you and never installs packages on your behalf.

You do **not** need to modify any metric or analysis code. Registering a potential is the only
edit required.

---

## The idea

Each component has a generator notebook containing a `POTENTIAL_REGISTRY` dictionary. Every entry
describes one FP: where its environment lives, which checkpoint to load, and the code that builds
its ASE calculator. The generator takes that entry and writes standalone run scripts, one per
job, which are then submitted on your cluster.

Adding an FP means adding one dictionary entry.

| Component | Generator notebook |
|---|---|
| Force prediction | `Force_error/generation/matpes_PBE_run_generator.ipynb` (and the `r2scan` / `omat24` variants) |
| Phase stability and ordering | `Phase_stability_ordering/generation/convexhull_ordering_run_generator.ipynb` |
| Ion migration by NEB | `Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb` |

---

## Registry fields

The three components grew separately and their field names differ slightly. Use the table for the
component you are working with, and copy the shape of an existing entry in that same notebook
rather than one from another component.

| Purpose | Force prediction | Phase stability / ordering | Ion migration by NEB |
|---|---|---|---|
| Name written into results | `fp_name` | `mlip_name` | `output_key` |
| Name shown in tables | (uses `fp_name`) | (uses `mlip_name`) | `display_name` |
| Environment activate script | `venv_activate` | `venv_activate` | `venv_activate` |
| Python executable | `python` | `python` | (not used) |
| site-packages directory | (not used) | `site_pkgs` | `site_pkgs` |
| Checkpoint path, or `None` | `model_path` | `model_path` | `model_path` |
| Calculator code | `calc_setup_code` | `calc_setup` | `import_lines` + `calc_lines` |

In the NEB component the calculator code is split in two, with imports in `import_lines` and the
calculator construction in `calc_lines`. In the other two it is a single block. In all cases the
code refers to the checkpoint through the variable `MODEL_PATH`, which the generator substitutes
from `model_path`.

The dictionary key itself (`"chgnet"`, `"mace"`, and so on) is the short registry key used for job
directory and submission script names. Keep it lowercase and filesystem-safe. Keys and result
names are asserted unique, so a collision fails loudly rather than overwriting another FP's
results.

---

## Example

The simplest case is an FP whose package ships its own weights, so `model_path` is `None`. This is
the CHGNet entry from the phase stability generator.

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

An FP that loads a checkpoint uses `MODEL_PATH` in its setup code.

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

The placeholder paths are not decorative. The generators check for values still starting with
`/path/to/` or `YOUR_` and raise a clear error rather than writing a job script that would fail
hours later on the cluster.

---

## Running it

1. **Register** your FP as above.
2. **Point the notebook at it.** Set the active potential or FP key list in the configuration
   section, and set the dataset, output, and SLURM settings for your system.
3. **Generate.** Run the generation cells. They write run scripts and submission scripts into
   your output directory. Nothing is submitted by running these cells.
4. **Submit** the generated scripts on your cluster.
5. **Merge.** When the jobs finish, run the merge section. It validates each job's output against
   the shared reference and writes a standardized results file.
6. **Analyse.** Open the component's analysis notebook, point it at your merged file, and it
   produces the same metric tables the leaderboard shows.

Steps 3 to 6 are documented per component in that component's README, under "Using FPBench" and
"Quick start".

---

## Things worth knowing

- **One environment per FP family.** This is the most common source of trouble. FP packages pin
  incompatible PyTorch and ASE versions, and every registry entry records its own environment
  precisely so they never have to share one.
- **Nothing is installed for you.** If a package is missing, the generated script raises a clear
  error rather than running `pip install`.
- **Generation is separate from submission.** Running the generator cells never launches a job.
- **Existing results are not overwritten.** Registry keys and result names are asserted unique.
- **Results stay tied to a model version.** When you report a benchmark, record the exact
  checkpoint. A later version of the same architecture is a different entry, not an update to an
  old one.

---

## Using FPBench on your own dataset instead

If you already have DFT and FP results and only want the metrics, you do not need the generators
at all. Each component's analysis functions are importable from its `scripts/` directory and
accept standardized inputs directly. See the "Using FPBench" and "Required inputs and outputs"
sections of the component README.
