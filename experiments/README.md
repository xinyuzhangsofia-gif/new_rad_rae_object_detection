# Experiment families

- `target_drop/`: primary weather Source-vs-Target definitions and Target Drop tables.
- `distance_ranges/`: fixed 0–30, 30–60, 60–90, and 90–120 metre evaluation.
- `distance_quartiles/`: equal-count GT-distance quartile and relative Target Drop analysis.
- `source_drop/`: controlled source-domain evaluation and Source Drop results.

Generated reports, logs, TensorBoard events, locks, and state retain each
family's existing internal layout. Local state or lock files stored beside a
table must move with that table; the repository does not provide duplicate
legacy directories or symlinks.
