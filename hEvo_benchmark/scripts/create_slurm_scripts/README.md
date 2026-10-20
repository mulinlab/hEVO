# Generate Slurm jobs

These tools generate Slurm job scripts. Run from the repository root
with Python >=3.10, PyYAML, Jinja2 and Rich. Configure checkpoints and data through
[project paths](../../configs/project.yaml), [model definitions](../../configs/models.yaml)
and [task definitions](../../configs/tasks.yaml), included by `hEvo_benchmark/config.yaml`.

## Entry points

| Tool | Workflow |
|---|---|
| `create_probe_single_seq.py` | Frozen-backbone classification; unified evaluator selects Pareto for NT_Pareto and standard otherwise |
| `create_zeroshot_paired_seq.py` | Paired-variant evaluation; native model-specific protocols |
| `create_popu_classification.py` | Sample embedding extraction for ancestry analysis |
| `gpn_star/` | Dedicated GPN-Star Slurm job scripts, including array and multi-GPU workflows |
| `split_config.py`, `migrate_config_paths.py` | Configuration maintenance |
| `config_utils.py`, `cli_utils.py` | Shared configuration/path resolution and console output |

CLI values take precedence over `CliDefaults`, which take precedence over local
fallbacks. Relative source configuration includes resolve against the config
file. For probe jobs, `--protocol` overrides the configured protocol; `auto`
selects Pareto for `NT_Pareto` and standard for other task groups. An explicit
`--evaluate-script` takes precedence over protocol selection and is called
without a `--protocol` argument. Older configs that specify `evaluate_script`
continue to use that path when no CLI protocol is given. Specify `--supername`
when several task groups exist. Unknown task names produce an error listing
valid groups.

```bash
python hEvo_benchmark/scripts/create_slurm_scripts/create_probe_single_seq.py \
  --supername NT_Pareto --python-exec "python" \
  --config-path hEvo_benchmark/config.yaml \
  --sbatch-root "$JOB_LOG_DIR" --slurm-root "$JOB_SCRIPT_DIR" \
  --results-root "$RUN_DIR" --hdf5-root "$EMBEDDING_CACHE_DIR" \
  --log-root "$PROBE_LOG_DIR"

python hEvo_benchmark/scripts/create_slurm_scripts/create_zeroshot_paired_seq.py \
  --supername ClinVar260615 --python-exec "python" \
  --config-path hEvo_benchmark/config.yaml \
  --sbatch-root "$JOB_LOG_DIR" --slurm-root "$JOB_SCRIPT_DIR" \
  --results-root "$RUN_DIR" --h5-root "$SCORE_CACHE_DIR" \
  --log-root "$PROBE_LOG_DIR"
```

Set these variables to writable directories before running. The generators create
script, log, and output directories. Inspect generated scripts and confirm
the site's partition, GPU, RAM, time and Python environment before using `sbatch`.
Use `--protocol pareto` to select the Pareto evaluator for Fig. 5
resource results, or `--evaluate-script` to select a custom or legacy evaluator.

GPU evaluation requires each model's runtime, checkpoint and input data. GPN-MSA
requires `GPN_MSA_PATH`; GPN-Star requires the paired store/species environment
variables shown in generated jobs. Missing or mismatched MSA stores fail at job
startup. Dataset column definitions must match task configuration.

[Evaluation guide](../evaluate/README.md) · [Ancestry guide](../population_classify/README.md)
