import pandas as pd
from pathlib import Path

# Load existing model outputs
df = pd.read_csv("eval/model_outputs.csv")

# Build manual evaluation template
template = pd.DataFrame({
    "query": df["query"],
    "answer": df["answer"],
    "reference": df["reference"],
    "relevance_1to5": "",
    "accuracy_1to5": "",
    "fluency_1to5": "",
    "source_present_yes_no": "",
    "notes": ""
})

# Save it
Path("eval").mkdir(exist_ok=True)
template.to_csv("eval/manual_eval.csv", index=False)
print("✅ Created eval/manual_eval.csv")
