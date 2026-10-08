# Experiment families

- `target_drop/`: primary weather Source-vs-Target definitions and Target Drop tables.
- `distance_quartiles/`: equal-count GT-distance quartile and relative Target Drop analysis.
- `controlled_splits/`: reproducibility artifacts for current optional controlled Source training.

Only controlled splits whose request signatures match current Target Drop
Source requests are retained. Complete retained directories include the frame
manifests, request metadata, object-ignore payload, statistics, and comparison
report. Existing legacy `controled_*` directories remain reusable through
signature matching; newly generated directories use the corrected
`controlled_*` prefix.

Removed historical experiment families remain available through Git history.

Generated reports, logs, TensorBoard events, locks, and state retain each
family's existing internal layout. Local state or lock files stored beside a
table must move with that table; the repository does not provide duplicate
legacy directories or symlinks.
