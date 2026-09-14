# Experiment families

- `target_drop/`: primary weather Source-vs-Target definitions and Target Drop tables.
- `distance_ranges/`: archived fixed-range reports; the evaluator was removed.
- `distance_quartiles/`: equal-count GT-distance quartile and relative Target Drop analysis.
- `source_drop/`: archived historical Source Drop results and control assets; the evaluator was removed.

Generated reports, logs, TensorBoard events, locks, and state retain each
family's existing internal layout. Local state or lock files stored beside a
table must move with that table; the repository does not provide duplicate
legacy directories or symlinks.
