# Batch and distributed runs

Use Python 3.12, Bash 4 or newer, and an activated environment containing the
selected backend. The smoke driver also uses GNU `timeout`. Dataset stores, GPU
allocations, and pretrained resources must be provisioned for the installation.

## Workspace and repetitions

Set `OTBENCH_WORKSPACE` to the task workspace, or use the sharded launcher's
`--workspace`. The default is the invocation's current directory. Data,
checkpoints, logs, and scheduler state resolve beneath that workspace; `STORE`,
`OUTPUTS`, and `CHECKPOINTS` override individual roots. The registry and materialized
schemas determine supported method/task/dataset combinations.

| Task | Default repetition protocol | Launcher |
| --- | --- | --- |
| Generation | Independent checkpoint and sampling seeds 0, 1, 2 | `run_generation_all.sh` |
| Imputation | Paired train/selection/mask/impute seeds `(s,s,s,s)`, `s=0,1,2`; four mechanisms × rates 0.1, 0.3, 0.5 | `run_imputation_all.sh` |
| Prediction | One checkpoint seed and one prediction trial per invocation | `run_prediction_all.sh` |
| Prediction, three seeds | Sequential checkpoint seeds 0, 1, 2 | `run_prediction_seeds.sh` |

```bash
export OTBENCH_WORKSPACE=/path/to/work
bash scripts/run_generation_all.sh METHOD DATASET
bash scripts/run_imputation_all.sh METHOD DATASET
bash scripts/run_prediction_seeds.sh METHOD DATASET
```

`N_TRIALS` sets generation's default seed range; `SEEDS` selects explicit generation
or imputation seeds. `TRAIN_SEED` selects the checkpoint seed for a single
prediction invocation. Imputation `TRIALS` controls draws within each setting,
separate from independent seed repetitions: native samplers default to three draws,
deterministic methods to one.

`SKIP_EXISTING=1` reuses only provenance- and artifact-validated completed results.
`METRIC` selects metrics; `EXTRA` forwards additional task arguments. For imputation,
`SCENARIOS` and `RATES` select the missingness grid. The task CLI and method config
preserve each method's information-access contract.

## Sharded runs and status

```bash
bash scripts/run_sharded.sh --config scripts/sharded.example.toml \
  --workspace /path/to/work --run-id experiment-001
python scripts/sharded_status.py --run-id experiment-001 --strict
```

Each launch freezes its selected shards under
`outputs/sharded/<task>/<method>/<run_id>/`. A run ID is unique; an existing ID is
rejected. Omitted IDs use UTC time plus a random suffix. Concurrent subsets use
distinct IDs. `--strict` requires every shard in the frozen selection to succeed
and reports selected/total coverage. Running, missing, skipped, or stale work is
not completion, and strict verification requires a valid frozen plan.

The example uses one GPU; adjust resource fields to the allocation. For prediction,
run train seeds 0, 1, and 2 as explicit launches or use the unsharded seed wrapper.
`--help` documents flags, configuration precedence, and status filters.

## Scheduler launches

`launch_rank_eval.sh` runs independent method/task workers. Supply `NNODES`,
`NODE_RANK` (or scheduler `RANK`), and one shared unique `RUN_ID`; allocate enough
ranks for the registry-derived plan. All nodes must see the same workspace and
frozen plan. Inspect completion with:

```bash
python scripts/rank_status.py --run-id experiment-001 --strict
```

The node entrypoint can finish successfully while a worker failed, to avoid a gang
scheduler killing healthy nodes. Status files and strict validation determine the
run verdict. Unassigned reserve ranks must be idle; expected work must succeed.

`launch_limix2_imputation.sh` uses torchrun to assign independent dataset/seed units;
the workers do not form a distributed model process group. It requires scheduler
rendezvous variables plus one shared unique `RUN_ID`. Both launchers use
the active environment; optional conda initialization requires explicit `CONDA_SH`
and `ENV_NAME`. Validate real multi-node execution on the intended shared
storage/network allocation; unit tests and plan-only checks do not establish it.
