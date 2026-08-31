# Benchmark Results: python-structural-retrieval-v1

## Methodology

- Language: Python only
- Queries: 24 frozen labeled cases (6 lexical / 6 behavioral / 6 calls / 6 inheritance)
- Top-K: 10
- Metrics: Hit@1, Hit@5, Hit@10, MRR@10 (ranking only; scores not compared across modes)
- Modes (fixed order): lexical, dense, hybrid, graph, reranked
- No ranking-constant tuning on this evaluation set
- Benchmark SHA-256: `5125c8facaa3344417ca5ea31958f2f6d1a1393a3675963c8e2cbf8d611ec2de`

## Setup

- Engine version: `0.1.0`
- Python: `3.14.7`
- Platform: `darwin`
- Provider: `sentence-transformers`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Corpus fingerprint: `1ac8a3a8dfc3d0bcf6840bfa229a7e44db9512ecdade9a54eeaa7a5ff548d8ef`

## Aggregate metrics

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.6667 | 0.8750 | 0.8750 | 0.7431 |
| dense | 0.7917 | 0.8750 | 1.0000 | 0.8532 |
| hybrid | 0.6667 | 0.8750 | 0.9583 | 0.7604 |
| graph | 0.2500 | 0.8333 | 0.9583 | 0.4971 |
| reranked | 0.2500 | 0.7500 | 0.9167 | 0.4241 |

## Category breakdown

### lexical

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| graph | 0.1667 | 0.8333 | 1.0000 | 0.4544 |
| reranked | 0.1667 | 0.5000 | 0.8333 | 0.3042 |

### behavioral

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.6667 | 1.0000 | 1.0000 | 0.7639 |
| dense | 0.6667 | 0.8333 | 1.0000 | 0.7778 |
| hybrid | 0.5000 | 1.0000 | 1.0000 | 0.6944 |
| graph | 0.1667 | 0.6667 | 1.0000 | 0.3396 |
| reranked | 0.1667 | 0.6667 | 0.8333 | 0.2935 |

### calls

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.5000 | 0.8333 | 0.8333 | 0.6250 |
| dense | 0.5000 | 0.6667 | 1.0000 | 0.6349 |
| hybrid | 0.5000 | 0.8333 | 1.0000 | 0.6528 |
| graph | 0.1667 | 1.0000 | 1.0000 | 0.5278 |
| reranked | 0.1667 | 1.0000 | 1.0000 | 0.4361 |

### inheritance

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| lexical | 0.5000 | 0.6667 | 0.6667 | 0.5833 |
| dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 0.6667 | 0.6667 | 0.8333 | 0.6944 |
| graph | 0.5000 | 0.8333 | 0.8333 | 0.6667 |
| reranked | 0.5000 | 0.8333 | 1.0000 | 0.6627 |

## Pairwise comparisons

- **graph vs hybrid**: wins=3, ties=7, losses=14
- **reranked vs graph**: wins=2, ties=7, losses=15

Missing ranks (no relevant hit in top-10) are treated as worse than any rank 1..10; both missing counts as a tie.

## Per-query first-relevant ranks

| ID | Category | Lexical | Dense | Hybrid | Graph | Reranked | Gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| lex-01 | lexical | 1 | 1 | 1 | 1 | 1 | `users.normalize_email` |
| lex-02 | lexical | 1 | 1 | 1 | 2 | 3 | `inventory.InventoryItem.has_stock` |
| lex-03 | lexical | 1 | 1 | 1 | 3 | 5 | `pricing.apply_tax_cents` |
| lex-04 | lexical | 1 | 1 | 1 | 7 | — | `payments.refund_payment` |
| lex-05 | lexical | 1 | 1 | 1 | 2 | 6 | `store.list_order_ids` |
| lex-06 | lexical | 1 | 1 | 1 | 4 | 8 | `orders.OrderLine` |
| beh-01 | behavioral | 1 | 1 | 1 | 1 | 1 | `users.normalize_email` |
| beh-02 | behavioral | 1 | 1 | 1 | 9 | — | `inventory.restock_units` |
| beh-03 | behavioral | 3 | 1 | 2 | 3 | 4 | `pricing.line_total_cents` |
| beh-04 | behavioral | 1 | 1 | 1 | 4 | 5 | `notifications.notify_customer_welcome` |
| beh-05 | behavioral | 4 | 2 | 3 | 7 | 9 | `store.delete_order` |
| beh-06 | behavioral | 1 | 6 | 3 | 5 | 5 | `orders.empty_order` |
| calls-01 | calls | — | 2 | 6 | 2 | 2 | `validation.reserve_order_inventory` |
| calls-02 | calls | 2 | 1 | 2 | 3 | 5 | `store.save_order` |
| calls-03 | calls | 1 | 1 | 1 | 1 | 1 | `notifications.notify_order_confirmed` |
| calls-04 | calls | 1 | 6 | 1 | 2 | 3 | `users.is_active_customer` |
| calls-05 | calls | 4 | 7 | 4 | 3 | 4 | `payments.authorize_card` |
| calls-06 | calls | 1 | 1 | 1 | 2 | 3 | `notifications.build_confirmation_subject` |
| inh-01 | inheritance | 1 | 1 | 1 | 1 | 1 | `policies.ChargePolicy` |
| inh-02 | inheritance | — | 1 | — | — | 7 | `policies.StrictChargePolicy` |
| inh-03 | inheritance | 1 | 1 | 1 | 2 | 3 | `policies.PrepaidChargePolicy` |
| inh-04 | inheritance | 2 | 1 | 1 | 1 | 2 | `discounts.DiscountPolicy` |
| inh-05 | inheritance | — | 1 | 6 | 2 | 1 | `discounts.PercentageDiscount` |
| inh-06 | inheritance | 1 | 1 | 1 | 1 | 1 | `discounts.LoyaltyDiscount` |

## Showcase cases (preselected before first real run)

### `lex-01` (lexical)

- Query: normalize_email helper
- Gold: `users.normalize_email`
- Hybrid rank: 1
- Graph rank: 1
- Reranked rank: 1
- Notes: Preselected lexical showcase before first real run.

### `calls-02` (calls)

- Query: where does checkout persist the order after a successful charge
- Gold: `store.save_order`
- Hybrid rank: 2
- Graph rank: 3
- Reranked rank: 5
- Notes: Preselected structural showcase before first real run; CALLS from finalize_checkout.

## Limitations

- Small synthetic committed corpus; not a claim about all codebases.
- Hand-authored queries and labels; author familiarity bias is possible.
- MiniLM is a general text embedding model, not code-specialized.
- Results are scoped to this benchmark only.
