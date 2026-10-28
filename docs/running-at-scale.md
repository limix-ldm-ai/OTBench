# Running at scale

[The launcher reference](../scripts/README.md) documents commands, environment
variables, repetition protocols, and status tools. Start with a single offline
task run before allocating a batch or multi-node job.

Use a shared workspace for the allocation. Keep source/configs separate from
`tabdata/store`, `checkpoints`, and `outputs`; `--workspace` or `OTBENCH_WORKSPACE`
relocates those runtime roots. `scripts/sharded.example.toml` describes one GPU;
set resource fields for the actual deployment.

Each sharded launch has an independent run ID and a frozen selected-shard plan
under `outputs/sharded/<task>/<method>/<run_id>/`. Omitted IDs use UTC time plus a
random UUID suffix. An existing ID is rejected. Concurrent subsets and
checkpoint-seed launches use different IDs and independent scheduler state.

`sharded_status.py --strict` checks all shards selected in the frozen launch and
shows selected/total coverage. `rank_status.py --strict` validates every expected
method/task assignment and its run/plan identity. Running, planned, missing,
skipped, stale, malformed, and identity-mismatched work is not successful
completion. A valid frozen plan is required for strict verification.

Rank workers are independent jobs. LimiX-2's torchrun launcher distributes
independent dataset/seed units and requires scheduler rendezvous. A successful
node entrypoint can coexist with failed rank workers to keep healthy gang-scheduled
nodes alive, so inspect strict run status rather than the node exit code alone.

Plan-only checks and unit tests do not establish GPU training quality, sufficient
device memory, shared-filesystem behavior, NCCL connectivity, or real multi-node
completion. Record the deployment and retain its frozen plan, status files, and
logs with the experiment artifacts.
