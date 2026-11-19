# FEAT-0019 · Capture Filter membership and measurement frames during execution

Record the parent, ordered root indices, resolved Filter parameters, output and DAG paths, role, and observed frame whenever a computation consumes a particle view. Record each named measurement frame. [Implementation plan](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

On the Phase 1 baseline ten tests reached the scientific values but failed because the Result had no selection observations. The joint and nested aperture controls were 28 and 4 kpc, with masses 3 and 2 Msol.

The full run now records detached origins, output roles and measurement frames. The runtime also identifies transient native transforms in a science hook and retains a shared selection target once across ten outputs.

[tests/test_selection_observations.py](../../../tests/test_selection_observations.py) · [Handoff](handoff.md)
