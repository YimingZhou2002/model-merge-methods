# Equal-norm temporal development round

57 tasks x8, FP32, T=0.9, seed20261112, scheduler4096, memory0.80.

| Model | Mean@8 % | Pass@8 % | Delta vs TV pp | Family 95% pp |
|---|---:|---:|---:|---|
| tsg_tv | 75.877 | 96.491 | reference |  |
| ck280 | 76.535 | 92.982 | +0.658 | [-1.754, 3.07] |
| tsg_tv_norm | 77.193 | 92.982 | +1.316 | [-1.974, 4.605] |
| bcm_linear_nm | 76.316 | 92.982 | +0.439 | [-2.851, 3.509] |
| qis_nm | 73.465 | 94.737 | -2.412 | [-6.579, 1.535] |
| drr_nm | 76.096 | 94.737 | +0.219 | [-2.851, 3.289] |
| bcm_relative_nm | 77.851 | 94.737 | +1.974 | [-1.096, 5.702] |

Pending: 

Development comparisons are unadjusted and conditional on these sessions.
