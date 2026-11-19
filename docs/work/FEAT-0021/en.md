# FEAT-0021 · Navigate and replay observed selection frames

Find an origin by Filter, declared transform or witnessed context. Replay supported operations on an independent particle view in the observed units and order. [Implementation plan](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

Four end-to-end cases reached the missing replay implementation. A real Gadget HDF snapshot also exposed a pynbody copy-on-access incompatibility with pathlib filenames. Verification compares to position, velocity and root indices copied at measurement time.

Verified native and selection-frame coordinates against the original measured values. The data-backed Gadget case retains the ordered indices [3, 1, 3, 0]. A custom loader can share a live root with other callers while replay uses independent arrays and units.

[tests/test_member_frames.py](../../../tests/test_member_frames.py) · [Handoff](handoff.md)
