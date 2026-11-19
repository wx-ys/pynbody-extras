# FEAT-0023 · Join subhalo, host and Study measurements by scientific locator

Export the complete catalogue, preserve unknown membership and missing measurements, and query objects by indexed typed ID and host. [Implementation plan](../../design/2026-11-19-pipeline-membership-and-result-queries-plan.zh-CN.md)

Three initial catalogue workflows failed at explicit object registration or paginated export. A subsequent test showed that global discovery omitted known hosts. The five-object inventory and fixed-size pages for 100 and 1,000 entries exercise the public API.

A five-row inventory retained measured zeroes, missing measurements and unknown membership through the subhalo and host joins. Indexed lookups and fixed pages used the same number of SQL queries for 100 and 1,000 objects. Mixed integer and floating-point catalogue IDs round-trip exactly.

[tests/test_database_member_catalogue_workflows.py](../../../tests/test_database_member_catalogue_workflows.py) · [Handoff](handoff.md)
