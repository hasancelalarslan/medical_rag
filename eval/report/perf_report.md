# Medical RAG Perf

**Input CSV:** `/home/arslanhasancelal/medical_rag/eval/perf_results.csv`

## Overall Latency Summary

|                    |   count |     mean |        std |    min |    max |     p50 |     p75 |     p90 |      p95 |      p99 |    p99.9 |
|:-------------------|--------:|---------:|-----------:|-------:|-------:|--------:|--------:|--------:|---------:|---------:|---------:|
| retrieval_time_ms  |     100 |   9.4685 |   0.163996 |   9.15 |  10.25 |   9.445 |   9.52  |   9.651 |   9.7715 |  10.0223 |  10.2272 |
| generation_time_ms |     100 | 336.222  | 147.137    | 118.53 | 974.18 | 320.92  | 400.205 | 472.659 | 528.437  | 933.62   | 970.124  |
| total_time_ms      |     100 | 348.156  | 147.134    | 130.41 | 986.38 | 332.73  | 412.115 | 484.701 | 540.356  | 945.681  | 982.31   |

## Latency by Query Length

| bucket   |   retrieval_time_ms_mean |   retrieval_time_ms_p95 |   generation_time_ms_mean |   generation_time_ms_p95 |   total_time_ms_mean |   total_time_ms_p95 |
|:---------|-------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|--------------------:|
| Long     |                  9.51971 |                   9.806 |                   348.973 |                  507.45  |              360.977 |             519.542 |
| Medium   |                  9.44114 |                   9.695 |                   348.944 |                  663.109 |              360.853 |             675.094 |
| Short    |                  9.44067 |                   9.573 |                   306.505 |                  481.749 |              318.386 |             493.654 |

## Top 20 Slowest Queries

| query                                                                      | query_length   |   retrieval_time_ms |   generation_time_ms |   total_time_ms |
|:---------------------------------------------------------------------------|:---------------|--------------------:|---------------------:|----------------:|
| Explain the use of statins in cardiovascular disease.                      | Medium         |                9.56 |               974.18 |          986.38 |
| How should antiretroviral therapy be modified in drug-resistant HIV?       | Long           |                9.4  |               933.21 |          945.27 |
| What are the current treatment options for type 2 diabetes?                | Medium         |                9.32 |               903.44 |          915.18 |
| Describe eclampsia.                                                        | Short          |                9.5  |               625.05 |          637.01 |
| What are the guidelines for cervical cancer screening?                     | Medium         |                9.35 |               560.11 |          572.2  |
| Describe the post-operative care after coronary artery bypass surgery.     | Long           |                9.37 |               526.77 |          538.68 |
| How is Crohn's disease diagnosed?                                          | Medium         |                9.47 |               516.12 |          527.85 |
| Define metabolic syndrome.                                                 | Short          |                9.42 |               506    |          517.77 |
| Describe the stepwise pharmacologic therapy for heart failure.             | Long           |                9.54 |               499.17 |          511.34 |
| What are the dietary recommendations for chronic kidney disease?           | Long           |                9.44 |               472.92 |          484.71 |
| What are the ethical considerations in end-of-life care?                   | Long           |                9.66 |               472.63 |          484.7  |
| What is migraine?                                                          | Short          |                9.46 |               452.11 |          464.18 |
| How should biologic therapies be used in rheumatoid arthritis?             | Long           |                9.26 |               448.4  |          460.19 |
| What are the strategies for reducing hospital readmission in COPD?         | Long           |                9.54 |               444.59 |          456.73 |
| Explain kidney stones.                                                     | Short          |                9.15 |               443.46 |          454.85 |
| What are the complications of hypertension?                                | Medium         |                9.27 |               438.76 |          450.44 |
| How should acute ischemic stroke be managed within the first 24 hours?     | Long           |                9.43 |               436.21 |          448.03 |
| What are the cancer screening methods?                                     | Medium         |                9.55 |               434.03 |          445.95 |
| What is cataract?                                                          | Short          |                9.45 |               431.85 |          443.77 |
| How should lifestyle changes be implemented in patients with hypertension? | Long           |                9.42 |               430.24 |          442.03 |

