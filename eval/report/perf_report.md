# Medical RAG Perf — Baykar Submission

**Input CSV:** `/app/eval/perf_results.csv`

## Overall Latency Summary

|                    |   count |      mean |       std |    min |      max |      p50 |      p75 |      p90 |       p95 |       p99 |      p99.9 |
|:-------------------|--------:|----------:|----------:|-------:|---------:|---------:|---------:|---------:|----------:|----------:|-----------:|
| retrieval_time_ms  |     100 |   27.6083 |   11.3074 |  13.9  |    56.09 |   23.035 |   34.145 |   46.589 |   49.9605 |   55.6148 |    56.0425 |
| generation_time_ms |     100 | 3526.88   | 3426.78   | 961.11 | 35371.6  | 3062.62  | 3928.34  | 4828.27  | 5232.65   | 9243.56   | 32758.8    |
| total_time_ms      |     100 | 3558.57   | 3426.36   | 987.34 | 35392.6  | 3104.4   | 3960.83  | 4849.54  | 5257.16   | 9298.29   | 32783.2    |

## Latency by Query Length

| bucket   |   retrieval_time_ms_mean |   retrieval_time_ms_p95 |   generation_time_ms_mean |   generation_time_ms_p95 |   total_time_ms_mean |   total_time_ms_p95 |
|:---------|-------------------------:|------------------------:|--------------------------:|-------------------------:|---------------------:|--------------------:|
| Long     |                  28.8509 |                 50.128  |                   3025.14 |                  5058.05 |              3058.14 |             5080.51 |
| Medium   |                  26.8531 |                 49.711  |                   4095.82 |                  6389.7  |              4126.97 |             6418.6  |
| Short    |                  27.0397 |                 45.2345 |                   3448.47 |                  5499.21 |              3479.28 |             5527.42 |

## Top 20 Slowest Queries

| query                                                                     | query_length   |   retrieval_time_ms |   generation_time_ms |   total_time_ms |
|:--------------------------------------------------------------------------|:---------------|--------------------:|---------------------:|----------------:|
| What are the ways to prevent influenza?                                   | Medium         |               15.81 |             35371.6  |        35392.6  |
| Explain rehabilitation after myocardial infarction.                       | Medium         |               49.97 |              8979.64 |         9034.71 |
| What is HIV?                                                              | Short          |               21.89 |              6111.21 |         6136.69 |
| Define stroke.                                                            | Short          |               26.93 |              5719.34 |         5750.27 |
| What are the cancer screening methods?                                    | Medium         |               13.96 |              5279.72 |         5297.41 |
| Explain multiple sclerosis.                                               | Short          |               21.04 |              5230.17 |         5255.04 |
| What are the ethical considerations in end-of-life care?                  | Long           |               32.07 |              5170.05 |         5206.35 |
| How should acute ischemic stroke be managed within the first 24 hours?    | Long           |               23.63 |              5130.33 |         5157.71 |
| What is diabetes?                                                         | Short          |               25.44 |              5121.63 |         5151.09 |
| How should depression be managed in patients with chronic illness?        | Long           |               15.42 |              5027.08 |         5047.42 |
| What is hepatitis C?                                                      | Short          |               17.93 |              4806.18 |         4827.55 |
| What are the complications of lupus?                                      | Medium         |               17.03 |              4764.36 |         4785.11 |
| Explain kidney stones.                                                    | Short          |               48.38 |              4631.74 |         4685.14 |
| How should cancer pain be assessed and managed?                           | Long           |               19.89 |              4501.6  |         4525.63 |
| What are the device therapies available for advanced stage heart failure? | Long           |               49.96 |              4437.69 |         4492.09 |
| What is the therapy for endometriosis?                                    | Medium         |               17.69 |              4276.82 |         4297.91 |
| What is influenza?                                                        | Short          |               20.78 |              4235.91 |         4260.53 |
| What are the treatment methods for iron deficiency anemia?                | Medium         |               25.64 |              4202.61 |         4233.5  |
| Describe non-pharmacologic interventions for dementia care.               | Long           |               27.33 |              4156.89 |         4188.53 |
| How should biologic therapies be used in rheumatoid arthritis?            | Long           |               49.58 |              4115.28 |         4170.58 |

