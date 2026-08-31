# Incremental Work Evaluation v1

Exact M8 work counters comparing incremental transitions vs full rebuilds on a
temporary copy of `benchmarks/python_retrieval_v1/corpus`.

- Engine version: `0.1.0`
- Corpus source: `benchmarks/python_retrieval_v1/corpus`
- Dense provider note: Dense work counts use FakeEmbeddingProvider (fake-incremental-work). They measure reuse/embed work, not retrieval quality.

No wall-clock timing claims.

## no-op

Unchanged corpus after baseline index.

| Counter | Incremental | Full |
| --- | ---: | ---: |
| mode | noop | full |
| files_analyzed | 0 | 11 |
| relation_files_recomputed | 0 | 11 |
| symbols_rewritten | 0 | 64 |
| code_units_rewritten | 0 | 53 |
| files_added | 0 | 11 |
| files_changed | 0 | 0 |
| files_deleted | 0 | 0 |
| files_unchanged | 11 | 0 |

| Dense counter | Selective embed | Full embed |
| --- | ---: | ---: |
| documents_total | 53 | 53 |
| vectors_reused | 53 | 0 |
| vectors_embedded | 0 | 53 |

Semantic equivalence of final CodeUnit qnames: `True`

## body-edit

Comment-only body change in pricing.line_total_cents.

| Counter | Incremental | Full |
| --- | ---: | ---: |
| mode | incremental | full |
| files_analyzed | 1 | 11 |
| relation_files_recomputed | 1 | 11 |
| symbols_rewritten | 4 | 64 |
| code_units_rewritten | 3 | 53 |
| files_added | 0 | 11 |
| files_changed | 1 | 0 |
| files_deleted | 0 | 0 |
| files_unchanged | 10 | 0 |

| Dense counter | Selective embed | Full embed |
| --- | ---: | ---: |
| documents_total | 53 | 53 |
| vectors_reused | 52 | 0 |
| vectors_embedded | 1 | 53 |

Semantic equivalence of final CodeUnit qnames: `True`

## symbol-rename

Rename pricing.line_total_cents → pricing.line_amount_cents.

| Counter | Incremental | Full |
| --- | ---: | ---: |
| mode | incremental | full |
| files_analyzed | 2 | 11 |
| relation_files_recomputed | 11 | 11 |
| symbols_rewritten | 14 | 64 |
| code_units_rewritten | 12 | 53 |
| files_added | 0 | 11 |
| files_changed | 2 | 0 |
| files_deleted | 0 | 0 |
| files_unchanged | 9 | 0 |

| Dense counter | Selective embed | Full embed |
| --- | ---: | ---: |
| documents_total | 53 | 53 |
| vectors_reused | 51 | 0 |
| vectors_embedded | 2 | 53 |

Semantic equivalence of final CodeUnit qnames: `True`

## add-delete

Add audit.py and delete payments.py.

| Counter | Incremental | Full |
| --- | ---: | ---: |
| mode | incremental | full |
| files_analyzed | 1 | 11 |
| relation_files_recomputed | 11 | 11 |
| symbols_rewritten | 2 | 62 |
| code_units_rewritten | 1 | 51 |
| files_added | 1 | 11 |
| files_changed | 0 | 0 |
| files_deleted | 1 | 0 |
| files_unchanged | 10 | 0 |

| Dense counter | Selective embed | Full embed |
| --- | ---: | ---: |
| documents_total | 51 | 51 |
| vectors_reused | 50 | 0 |
| vectors_embedded | 1 | 51 |

Semantic equivalence of final CodeUnit qnames: `True`
