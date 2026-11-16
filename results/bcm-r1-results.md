# Temporal merge search round 1 (development only)

All cells: 57 tasks x8, T=0.9, FP32, seed=20261110. Original dev has already been examined.

| Model | Mean@8 % | Pass@8 % | Delta vs concurrent TV pp | Family 95% pp |
|---|---:|---:|---:|---|
| tsg_tv | 75.658 | 94.737 | reference |  |
| bcm_linear | 77.851 | 96.491 | +2.193 | [-0.658, 5.044] |
| qis_05 | 72.807 | 91.228 | -2.851 | [-5.921, -0.219] |
| qis_10 | 76.096 | 94.737 | +0.439 | [-2.851, 3.509] |
| drr_05 | 75.658 | 98.246 | +0.000 | [-3.509, 3.509] |
| drr_10 | 75.000 | 92.982 | -0.658 | [-3.07, 1.754] |
| bcm_05 | 78.728 | 94.737 | +3.070 | [1.096, 5.263] |
| bcm_10 | 78.947 | 92.982 | +3.289 | [1.096, 5.702] |

Pending: 

Exploratory reused-dev search; unadjusted intervals conditional on these sessions. No confirmed winners yet.
