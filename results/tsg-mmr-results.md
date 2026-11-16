# Qwen3-14B AppWorld dev@8

T=0.9, FP32, 57 tasks x8, eight GPUs.

| Model | Mean@8 | Pass@8 |
|---|---:|---:|
| ck230 | 0.506579 | 0.824561 |
| ck240 | 0.570175 | 0.877193 |
| ck250 | 0.614035 | 0.859649 |
| ck260 | 0.627193 | 0.912281 |
| ck270 | 0.745614 | 0.947368 |
| ck280 | 0.767544 | 0.929825 |
| ck290 | 0.754386 | 0.964912 |
| ck300 | 0.736842 | 0.929825 |
| mmr_ties | 0.769737 | 0.964912 |
| linear | 0.780702 | 0.947368 |
| tsg_tv | 0.785088 | 0.964912 |
| tsg_tv_norm | 0.767544 | 0.947368 |

| Comparison | Delta (pp) | Primary bootstrap 95% interval (pp) | Method |
|---|---:|---|---|
| mmr_ties - ck300 | 3.289 | [-0.219, 7.018] | task-family clusters |
| mmr_ties - ck280 | 0.219 | [-5.263, 5.263] | task-family clusters |
| linear - ck300 | 4.386 | [1.754, 7.456] | task-family clusters |
| linear - ck280 | 1.316 | [-1.535, 4.167] | task-family clusters |
| tsg_tv - ck300 | 4.825 | [1.974, 7.675] | task-family clusters |
| tsg_tv - ck280 | 1.754 | [-2.851, 6.579] | task-family clusters |
| tsg_tv_norm - ck300 | 3.070 | [-0.658, 7.018] | task-family clusters |
| tsg_tv_norm - ck280 | 0.000 | [-3.070, 3.070] | task-family clusters |
| mmr_ties - linear | -1.096 | [-4.605, 2.412] | task-family clusters |
| tsg_tv - linear | 0.439 | [-2.412, 3.509] | task-family clusters |
| tsg_tv - tsg_tv_norm | 1.754 | [-2.632, 6.360] | task-family clusters |
| tsg_tv - mmr_ties | 1.535 | [-2.412, 5.482] | task-family clusters |

One session per model; source selection and merge assessment reuse dev. Best-source selection and unadjusted comparisons are exploratory, not independent validation.
Intervals resample task families when identified from frozen AppWorld task instructions, otherwise tasks; never rollout rows or independent sessions.
