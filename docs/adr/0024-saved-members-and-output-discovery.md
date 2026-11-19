# ADR-024 · Saved members and output discovery

2026-11-20 · [Member storage](../work/FEAT-0020/en.md) · [Frame replay](../work/FEAT-0021/en.md) · [Result discovery](../work/FEAT-0022/en.md) · [Catalogue joins](../work/FEAT-0023/en.md)

A catalogue describes scientific objects and their declared host relationships.
Several catalogue entries can share one exact particle input. ResultStore
continues to identify a computation by its exact input and CalculatorSignature.
A scientific locator joins inventory and result rows by the database namespace,
catalogue identity, object kind and typed original number. The catalogue
identity includes the root revision, finder, loader and catalogue revision.

Database stages the detached observations from the execution with SaveOperation.
It publishes the origins, output frames, version-specific uses and result summary
in the confirmation transaction. Recovery follows the recorded receipt.
A cache hit records its reuse Run and keeps the producer's member relationships.
A shared ordered particle sequence has one canonical Selection definition.
The same sequence can be observed under multiple parent objects and frames.
Independent explicit registration creates an origin without a saved result.

A member collection groups matching origins by exact input for direct loading.
A request for a particular selection frame resolves an origin and replays its
ordered, witnessed operations on independent arrays. The snapshot revision and
family layout must match. Old particle files and their values are maintained by
the data owner. [Runtime evidence](0023-selection-observations.md)

Discovery reads recipe metadata and selects a pinned version inside a database
read transaction. Predicates apply to one output path. Named Pipeline leaves
share the root receipt. Filters used to measure a transform carry a separate
role from filters on the measured output. A pre-observation recipe remains
discoverable from its stored signature. An unrecorded use list and an observed
empty list remain distinguishable. Fixed Study cells address historical
measurements through the existing database-qualified SavedResultRef.

[English guide](../database.md) · [中文指南](../database.zh-CN.md)
