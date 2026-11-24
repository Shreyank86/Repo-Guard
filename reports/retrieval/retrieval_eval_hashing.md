# Retrieval evaluation (hashing)

Benchmark: `retrieval_benchmark_v1.jsonl` (757 cases). Index ids: `cachetools@c5de5137153b-81e66c64`, `python-dotenv@4fbc6ff07215-81e66c64`, `python-tabulate@268615a5c27d-66f82ca2`, `requests@3c7baaa05ab4-81e66c64`, `schedule@82a43db1b938-66f82ca2`

Relevance unit = symbol (all parts of one function count once). `requests` is the held-out test repository.

## By repository and case type

| repository_id | case_type | n | hit@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_p50_ms | latency_p95_ms | empty_rate | mean_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cachetools | callers | 60 | 0.0167 | 0.5739 | 0.8189 | 0.2914 | 0.4238 | 0.99 | 1.318 | 0.0 | 534.3 |
| cachetools | docstring | 37 | 0.8649 | 0.973 | 1.0 | 0.9167 | 0.9373 | 0.67 | 1.134 | 0.0 | 468.2 |
| cachetools | symbol_lookup | 60 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.001 | 1.104 | 0.0 | 452.6 |
| python-dotenv | callers | 30 | 0.0667 | 0.775 | 0.8667 | 0.4168 | 0.5409 | 0.514 | 0.679 | 0.0 | 1019.2 |
| python-dotenv | docstring | 19 | 0.8421 | 0.9474 | 0.9474 | 0.886 | 0.9016 | 0.364 | 0.605 | 0.0 | 1189.8 |
| python-dotenv | symbol_lookup | 60 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.509 | 0.624 | 0.0 | 708.9 |
| python-tabulate | callers | 51 | 0.0588 | 0.9036 | 0.9706 | 0.4351 | 0.5778 | 0.929 | 1.037 | 0.0 | 1294.4 |
| python-tabulate | docstring | 37 | 0.7297 | 0.9189 | 0.9189 | 0.8198 | 0.8456 | 0.647 | 1.101 | 0.0 | 1135.4 |
| python-tabulate | symbol_lookup | 60 | 0.9833 | 1.0 | 1.0 | 0.9917 | 0.9938 | 0.906 | 1.057 | 0.0 | 1211.9 |
| requests | callers | 60 | 0.0167 | 0.7019 | 0.93 | 0.3682 | 0.5244 | 1.372 | 1.6 | 0.0 | 1135.3 |
| requests | docstring | 60 | 0.7667 | 0.9167 | 0.9667 | 0.8378 | 0.8693 | 0.96 | 1.748 | 0.0 | 811.3 |
| requests | symbol_lookup | 60 | 0.9667 | 1.0 | 1.0 | 0.9833 | 0.9877 | 1.306 | 1.513 | 0.0 | 832.5 |
| schedule | baseline_q | 50 | 0.86 | 0.86 | 0.88 | 0.8722 | 0.8591 | 0.394 | 0.497 | 0.0 | 760.5 |
| schedule | callers | 27 | 0.0741 | 0.7043 | 0.9136 | 0.4272 | 0.5657 | 0.39 | 0.443 | 0.0 | 1042.8 |
| schedule | docstring | 26 | 0.6154 | 0.9615 | 0.9615 | 0.7558 | 0.8073 | 0.35 | 0.504 | 0.0 | 811.2 |
| schedule | symbol_lookup | 60 | 0.9833 | 1.0 | 1.0 | 0.9917 | 0.9938 | 0.408 | 0.544 | 0.0 | 843.3 |
| ALL | ALL | 757 | 0.6407 | 0.8929 | 0.9526 | 0.7641 | 0.8132 | 0.886 | 1.485 | 0.0 | 868.7 |


## By case type (all repositories)

| case_type | n | hit@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_p50_ms | latency_p95_ms | empty_rate | mean_tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_q | 50 | 0.86 | 0.86 | 0.88 | 0.8722 | 0.8591 | 0.394 | 0.497 | 0.0 | 760.5 |
| callers | 228 | 0.0395 | 0.7232 | 0.8996 | 0.3763 | 0.5169 | 0.972 | 1.524 | 0.0 | 986.5 |
| docstring | 179 | 0.7654 | 0.9385 | 0.9609 | 0.8436 | 0.8729 | 0.7 | 1.515 | 0.0 | 847.5 |
| symbol_lookup | 300 | 0.9867 | 1.0 | 1.0 | 0.9933 | 0.9951 | 0.912 | 1.423 | 0.0 | 809.8 |
| ALL | 757 | 0.6407 | 0.8929 | 0.9526 | 0.7641 | 0.8132 | 0.886 | 1.485 | 0.0 | 868.7 |


## Hybrid (default) by split

| split | case_type | n | hit@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_p50_ms | latency_p95_ms | empty_rate | mean_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | baseline_q | 50 | 0.86 | 0.86 | 0.88 | 0.8722 | 0.8591 | 0.394 | 0.497 | 0.0 | 760.5 |
| dev | callers | 168 | 0.0476 | 0.7309 | 0.8887 | 0.3792 | 0.5143 | 0.93 | 1.099 | 0.0 | 933.4 |
| dev | docstring | 119 | 0.7647 | 0.9496 | 0.958 | 0.8465 | 0.8747 | 0.607 | 1.083 | 0.0 | 865.8 |
| dev | symbol_lookup | 240 | 0.9917 | 1.0 | 1.0 | 0.9958 | 0.9969 | 0.81 | 1.06 | 0.0 | 804.2 |
| test | callers | 60 | 0.0167 | 0.7019 | 0.93 | 0.3682 | 0.5244 | 1.372 | 1.6 | 0.0 | 1135.3 |
| test | docstring | 60 | 0.7667 | 0.9167 | 0.9667 | 0.8378 | 0.8693 | 0.96 | 1.748 | 0.0 | 811.3 |
| test | symbol_lookup | 60 | 0.9667 | 1.0 | 1.0 | 0.9833 | 0.9877 | 1.306 | 1.513 | 0.0 | 832.5 |
| ALL | ALL | 757 | 0.6407 | 0.8929 | 0.9526 | 0.7641 | 0.8132 | 0.886 | 1.485 | 0.0 | 868.7 |


## Ablation: dense only (definition_boost = mention_boost = 0) by split

| split | case_type | n | hit@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_p50_ms | latency_p95_ms | empty_rate | mean_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | baseline_q | 50 | 0.4 | 0.55 | 0.6 | 0.4862 | 0.4903 | 0.295 | 0.393 | 0.0 | 545.3 |
| dev | callers | 168 | 0.1369 | 0.4809 | 0.6572 | 0.3294 | 0.4021 | 0.544 | 0.839 | 0.0 | 729.2 |
| dev | docstring | 119 | 0.7815 | 0.9496 | 0.958 | 0.8549 | 0.8809 | 0.541 | 1.077 | 0.0 | 837.9 |
| dev | symbol_lookup | 240 | 0.5708 | 0.7729 | 0.8479 | 0.6626 | 0.707 | 0.485 | 0.683 | 0.0 | 565.6 |
| test | callers | 60 | 0.1167 | 0.5222 | 0.6478 | 0.3546 | 0.4035 | 0.803 | 0.986 | 0.0 | 833.1 |
| test | docstring | 60 | 0.7833 | 0.9167 | 0.9667 | 0.8461 | 0.8754 | 0.833 | 1.05 | 0.0 | 799.9 |
| test | symbol_lookup | 60 | 0.65 | 0.85 | 0.8833 | 0.7353 | 0.7689 | 0.788 | 0.933 | 0.0 | 545.4 |
| ALL | ALL | 757 | 0.4835 | 0.7188 | 0.8029 | 0.6031 | 0.6465 | 0.56 | 0.908 | 0.0 | 681.6 |


## Ablation: exclude test files (`SearchFilters(exclude_tests=True)`)

| case_type | n | hit@1 | recall@5 | recall@10 | mrr | ndcg@10 | latency_p50_ms | latency_p95_ms | empty_rate | mean_tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_q | 50 | 0.86 | 0.86 | 0.89 | 0.8722 | 0.8626 | 0.782 | 1.031 | 0.0 | 748.2 |
| callers | 228 | 0.0439 | 0.7133 | 0.7974 | 0.361 | 0.4703 | 2.398 | 3.33 | 0.0 | 1062.0 |
| docstring | 179 | 0.7821 | 0.9609 | 0.9665 | 0.8584 | 0.8857 | 2.328 | 4.636 | 0.0 | 955.1 |
| symbol_lookup | 300 | 0.99 | 0.9983 | 0.9983 | 0.995 | 0.995 | 1.982 | 4.4 | 0.0 | 840.7 |
| ALL | 757 | 0.6473 | 0.8945 | 0.9231 | 0.7636 | 0.8024 | 2.04 | 4.387 | 0.0 | 928.3 |


## Free signals on the baseline question set (means)

| question_type | n | is_trap | symbol_found | name_similarity | retr_top1 | retr_margin | passage_mentions_symbol | stack_match | coverage | vocab_overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| callers | 10 | False | 0.9 | 0.995 | 0.51 | 0.167 | 1.0 | 1.0 | 0.4 | 1.0 |
| imports | 5 | False | 0.8 | 0.8 | 0.379 | 0.034 | 0.2 | 1.0 | 0.0 | 0.867 |
| inheritance | 3 | False | 1.0 | 1.0 | 0.305 | 0.064 | 1.0 | 1.0 | 1.0 | 0.778 |
| parameters | 15 | False | 1.0 | 1.0 | 0.24 | 0.049 | 1.0 | 1.0 | 1.0 | 0.987 |
| trap_near_miss | 15 | True | 0.0 | 0.708 | 0.199 | 0.02 | 0.0 | 1.0 | 0.0 | 0.624 |
| trap_nonexistent | 15 | True | 0.0 | 0.621 | 0.175 | 0.025 | 0.0 | 0.933 | 0.0 | 0.377 |
| trap_off_topic | 10 | True | 0.1 | 0.262 | 0.154 | 0.022 | 0.2 | 0.2 | 0.2 | 0.161 |
| trap_parameters | 10 | True | 0.0 | 0.63 | 0.237 | 0.032 | 1.0 | 1.0 | 1.0 | 0.697 |
| where_defined | 17 | False | 1.0 | 1.0 | 0.352 | 0.065 | 1.0 | 1.0 | 0.941 | 0.941 |


## Signal separation: AUROC answerable vs. each trap type (0.5 = none, 1.0 = perfect)

| answerable_vs | n_traps | symbol_found | name_similarity | retr_top1 | retr_margin | passage_mentions_symbol | stack_match | coverage | vocab_overlap |
|---|---|---|---|---|---|---|---|---|---|
| trap_nonexistent | 15 | 0.98 | 0.98 | 0.8547 | 0.7507 | 0.96 | 0.5333 | 0.88 | 0.9647 |
| trap_near_miss | 15 | 0.98 | 0.98 | 0.788 | 0.7947 | 0.96 | 0.5 | 0.88 | 0.9433 |
| trap_parameters | 10 | 0.98 | 0.98 | 0.72 | 0.74 | 0.46 | 0.5 | 0.38 | 0.798 |
| trap_off_topic | 10 | 0.93 | 0.936 | 0.89 | 0.776 | 0.86 | 0.9 | 0.78 | 0.981 |
