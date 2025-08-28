# Medical RAG Perf

**Input CSV:** `/home/arslanhasancelal/medical_rag/eval/perf_results.csv`

## Overall Latency Summary

|                    |   count |     mean |       std |   min |    max |    p50 |      p75 |     p90 |      p95 |      p99 |   p99.9 |
|:-------------------|--------:|---------:|----------:|------:|-------:|-------:|---------:|--------:|---------:|---------:|--------:|
| retrieval_time_ms  |     100 |   9.4856 |   1.23363 |  9.03 |  21.56 |   9.31 |   9.4525 |   9.601 |   9.7415 |  10.7195 |  20.476 |
| generation_time_ms |     100 | 289.079  | 119.384   | 67.49 | 893.02 | 273.48 | 325.358  | 429.022 | 497.788  | 664.191  | 870.137 |
| total_time_ms      |     100 | 301.21   | 119.552   | 79.56 | 904.76 | 285.49 | 337.19   | 440.705 | 511.015  | 676.753  | 881.959 |

## Latency by Query Length

| bucket   |   retrieval_time_ms_mean |   retrieval_time_ms_p95 |   generation_time_ms_mean |   generation_time_ms_p95 |   total_time_ms_mean |   total_time_ms_p95 |
|:---------|-------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|--------------------:|
| Long     |                  9.79714 |                  10.022 |                   336.321 |                  569.039 |              348.739 |             581.259 |
| Medium   |                  9.30914 |                   9.66  |                   246.507 |                  350.223 |              258.46  |             362.348 |
| Short    |                  9.328   |                   9.655 |                   283.631 |                  450.306 |              295.635 |             462.285 |

## Top 20 Slowest Queries

| query                                                                              | query_length   |   retrieval_time_ms |   generation_time_ms |   total_time_ms |
|:-----------------------------------------------------------------------------------|:---------------|--------------------:|---------------------:|----------------:|
| What are the management strategies for stroke rehabilitation?                      | Long           |                9.09 |               893.02 |          904.76 |
| What is the latest guideline for breast cancer screening in high-risk populations? | Long           |                9.58 |               661.88 |          674.45 |
| What is psoriasis?                                                                 | Short          |                9.45 |               553.51 |          565.62 |
| What are the recommendations for vaccination in immunocompromised patients?        | Long           |                9.45 |               529.25 |          541.32 |
| How should statin therapy be adjusted in elderly patients?                         | Long           |                9.22 |               507.44 |          519.28 |
| What are the ethical considerations in end-of-life care?                           | Long           |               10.61 |               497.28 |          510.58 |
| How should anticoagulation be managed in patients with atrial fibrillation?        | Long           |                9.6  |               493.83 |          505.87 |
| Describe gout.                                                                     | Short          |                9.55 |               468.27 |          480.51 |
| What are the updates in the management of severe sepsis?                           | Long           |                9.28 |               437.63 |          449.55 |
| What are the current recommendations for the management of chronic kidney disease? | Long           |                9.44 |               435.07 |          446.96 |
| Explain kidney stones.                                                             | Short          |                9.18 |               428.35 |          440.01 |
| How should anticoagulant reversal be performed in bleeding patients?               | Long           |                9.34 |               421.58 |          433.47 |
| What are the device therapies available for advanced stage heart failure?          | Long           |                9.77 |               408.19 |          420.55 |
| What are the indications for long-term oxygen therapy in patients with COPD?       | Long           |               21.56 |               394.84 |          419.24 |
| Define polycystic ovary syndrome.                                                  | Short          |                9.57 |               394.54 |          406.65 |
| What is Crohn's disease?                                                           | Short          |                9.49 |               392.73 |          404.78 |
| What is the stepwise approach to asthma control according to clinical guidelines?  | Long           |                9.58 |               392.35 |          404.54 |
| How is rheumatoid arthritis treated?                                               | Medium         |                9.44 |               390.87 |          402.94 |
| What are the strategies for reducing hospital readmission in COPD?                 | Long           |                9.46 |               379.95 |          392.07 |
| What is migraine?                                                                  | Short          |                9.42 |               365.14 |          377.43 |

