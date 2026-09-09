# Adding a Potential

FPBench evaluates a foundation potential (FP) through an ASE calculator. Adding an FP that is not
already on the leaderboard means describing, in one registry entry, how that calculator is built:
which environment to run it in, which checkpoint to load, and the few lines of Python that produce
a `calc` object. The existing component workflow does the rest.

No metric or analysis code requires modification. Registering the potential is the only edit
needed.

---

## What you need

- An **ASE-compatible calculator** for the FP, exposing `get_potential_energy()` and
  `get_forces()`.
- A **working Python environment** in which that calculator imports and runs.
- A **checkpoint or model weights**, where the FP requires one.
- **Access to a machine or cluster** able to run the calculations. FPBench generates the jobs; it
  does not execute them.

---

## Workflow

```
Register FP -> Generate calculations -> Run -> Merge results -> Analyze
```

1. **Register the potential.** Add one entry to the component's `POTENTIAL_REGISTRY`, as described
   below.
2. **Configure the notebook.** Set the active potential or FP key list in the configuration
   section, together with the dataset, output, and SLURM settings appropriate to the target
   system.
3. **Generate.** Running the generation cells writes run scripts and submission scripts into the
   output directory. No job is submitted at this stage.
4. **Run.** Submit the generated scripts on the cluster.
5. **Merge.** Once the jobs complete, the merge section validates each job's output against the
   shared reference and writes a standardized results file.
6. **Analyze.** The component's analysis notebook reads the merged file and produces the same
   metric tables reported on the leaderboard.

---

## Registering a potential

Each component contains a generator notebook holding a `POTENTIAL_REGISTRY` dictionary. Every
entry describes one FP: the environment it runs in, the checkpoint it loads, and the code that
constructs its ASE calculator. The generator expands that entry into standalone run scripts, one
per job, which are then submitted on the target cluster.

| Component | Generator notebook |
|---|---|
| Force prediction | `Force_error/generation/matpes_PBE_run_generator.ipynb`, and the `r2scan` and `omat24` variants |
| Phase stability and elemental ordering | `Phase_stability_ordering/generation/convexhull_ordering_run_generator.ipynb` |
| Ion migration by NEB | `Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb` |

Every registry records the same four quantities: the environment to run in, the checkpoint to
load, the code that constructs the ASE calculator, and the name written into the results. The
field names differ between components and are documented alongside the registry to which they
belong.

- [Force prediction, Registering a potential](Force_error/README.md#registering-a-potential)
- [Phase stability and ordering, Registering a potential](Phase_stability_ordering/README.md#registering-a-potential)
- [Ion migration by NEB, Registering a potential](Ion_migration_NEB/README.md#registering-a-potential)

A representative entry:

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

The calculator code refers to the checkpoint through the variable `MODEL_PATH`, which the
generator substitutes from the entry's `model_path`. Where a package supplies its own weights and
needs no path, set `model_path` to `None`. An existing entry in the same notebook should be used
as the template, rather than an entry from another component.

---

## Run a benchmark

Generation, submission, merging and analysis are documented per component, under "Using FPBench"
and "Quick start" in each README. Those READMEs and their notebooks are the source for detailed
operational instructions.

- [Force Prediction](Force_error/README.md)
- [Phase Stability & Ordering](Phase_stability_ordering/README.md)
- [Ion Migration (NEB)](Ion_migration_NEB/README.md)

---

## Using an independent dataset

Where DFT and FP results already exist and only the metrics are required, the generators are not
needed. Each component's analysis functions are importable from its `scripts/` directory and
accept standardized inputs directly. See the "Using FPBench" and "Required inputs and outputs"
sections of the component README.

---

## Public leaderboard

Everything above lets you evaluate any FP against the FPBench reference data on your own machine;
no permission or coordination is needed. Being listed on the public leaderboard is separate.

For a new FP to be considered for inclusion in the public leaderboard, please contact
Prof. Yifei Mo at <yfmo@umd.edu> with:

- model name
- version/checkpoint
- a link to the official implementation or model weights

---

## Technical notes

- **One environment per FP family.** FP packages pin incompatible PyTorch and ASE versions, and
  each registry entry records its own environment so that they need not be shared. A separate
  virtual environment per FP family is recommended rather than a single shared environment.
- **No automatic installation.** Where a package is missing, the generated script raises an
  explicit error rather than invoking `pip install`.
- **Generation is separate from submission.** Running the generator cells does not launch a job.
- **Placeholder paths are checked.** Values still beginning with `/path/to/` or `YOUR_` raise an
  explicit error rather than producing a job script that fails later on the cluster.
- **Existing results are preserved.** Registry keys and result names are asserted unique, so a
  collision fails explicitly rather than overwriting another FP's results.
- **Results are tied to a model version.** The exact checkpoint should be recorded when a
  benchmark is reported. A later version of the same architecture constitutes a separate entry
  rather than an update to an existing one.
