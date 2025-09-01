# Medical RAG Perf — Baykar Submission

**Input CSV:** `/home/arslanhasancelal/medical_rag/eval/perf_results.csv`

## Overall Latency Summary

|                    |   count |     mean |        std |   min |     max |     p50 |      p75 |     p90 |     p95 |       p99 |     p99.9 |
|:-------------------|--------:|---------:|-----------:|------:|--------:|--------:|---------:|--------:|--------:|----------:|----------:|
| retrieval_time_ms  |     100 |  10.0506 |   0.407543 |  9.57 |   11.55 |   9.925 |  10.2025 |  10.484 |  10.941 |   11.3718 |   11.5322 |
| generation_time_ms |     100 | 399.846  | 232.056    | 73.35 | 1652.3  | 348.875 | 499.223  | 624.996 | 716.504 | 1478.64   | 1634.93   |
| total_time_ms      |     100 | 412.497  | 232.067    | 85.92 | 1664.5  | 361.43  | 511.743  | 638.771 | 729.159 | 1491.34   | 1647.18   |

## Latency by Query Length

| bucket   |   retrieval_time_ms_mean |   retrieval_time_ms_p95 |   generation_time_ms_mean |   generation_time_ms_p95 |   total_time_ms_mean |   total_time_ms_p95 |
|:---------|-------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|--------------------:|
| Long     |                  10.0643 |                 10.734  |                   462.175 |                  753.983 |              474.875 |             766.677 |
| Medium   |                  10.0594 |                 11.192  |                   372.565 |                  680.886 |              385.185 |             694.524 |
| Short    |                  10.0243 |                 10.6265 |                   358.957 |                  691.982 |              371.585 |             704.425 |

## Top 20 Slowest Queries

| query                                                                              | query_length   |   retrieval_time_ms |   generation_time_ms |   total_time_ms |
|:-----------------------------------------------------------------------------------|:---------------|--------------------:|---------------------:|----------------:|
| What are the complications of hypertension?                                        | Medium         |                9.66 |              1652.3  |         1664.5  |
| Explain guideline-based management of gestational diabetes.                        | Long           |               10.01 |              1476.89 |         1489.59 |
| What are the recommendations for vaccination in immunocompromised patients?        | Long           |               10    |               895.74 |          908.42 |
| What is the first-line treatment for an asthma attack?                             | Medium         |               10.44 |               815.93 |          828.91 |
| What is hypertension?                                                              | Short          |               10.64 |               728.92 |          742.44 |
| What is cataract?                                                                  | Short          |                9.92 |               715.85 |          728.46 |
| How should anticoagulation be managed in patients with atrial fibrillation?        | Long           |               10.24 |               693.23 |          705.93 |
| Explain multidisciplinary care in amyotrophic lateral sclerosis.                   | Long           |                9.66 |               670.66 |          682.9  |
| What is tuberculosis?                                                              | Short          |                9.7  |               662.81 |          675.05 |
| What are the updates in the management of severe sepsis?                           | Long           |                9.9  |               642.87 |          655.34 |
| What are the guidelines for cervical cancer screening?                             | Medium         |               11.29 |               623.01 |          636.93 |
| How should pediatric asthma be treated according to the latest guidelines?         | Long           |               10.24 |               595.05 |          607.98 |
| What are the current recommendations for the management of chronic kidney disease? | Long           |               10.4  |               592.02 |          604.99 |
| How is palliative care integrated into cancer treatment plans?                     | Long           |                9.79 |               576.05 |          588.55 |
| What diseases are caused by obesity?                                               | Medium         |                9.99 |               563.04 |          575.57 |
| What are the ethical considerations in end-of-life care?                           | Long           |               10.31 |               562.11 |          575.02 |
| How should depression be managed in patients with chronic illness?                 | Long           |                9.77 |               550.23 |          562.56 |
| What are the surgical interventions for severe obesity?                            | Long           |                9.87 |               549.95 |          562.54 |
| How should cancer pain be assessed and managed?                                    | Long           |                9.8  |               541.21 |          553.75 |
| Define stroke.                                                                     | Short          |               10.47 |               526.2  |          539.27 |

