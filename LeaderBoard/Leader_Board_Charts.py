import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing model outputs
base_dir = r"D:\Video_Analysis\Output"

# Models to evaluate
models = ["Gemini-2.0-Flash", "Gemini-1.5-Flash", "Gemini-1.5-Pro"]
metrics = ["Correct", "Incorrect", "Don't Know"]

# Initialize counts
results = {model: {metric: 0 for metric in metrics} for model in models}

# Traverse each subfolder
for subdir, dirs, files in os.walk(base_dir):
    if "question.json" in files and "Answers_By_LLM.json" in files:
        q_path = os.path.join(subdir, "question.json")
        a_path = os.path.join(subdir, "Answers_By_LLM.json")
        
        with open(q_path, 'r', encoding='utf-8') as qf:
            questions = json.load(qf)
        with open(a_path, 'r', encoding='utf-8') as af:
            answers = json.load(af)
        
        # Map question numbers to correct answers
        correct_map = {q["Question_No."]: q["Answer"].strip() for q in questions}
        
        # Compare predictions
        for ans in answers:
            q_no = ans["Question_No."]
            correct = correct_map.get(q_no)
            for model in models:
                key = f"{model}_Answer"
                pred = ans.get(key, "").strip()
                
                if pred.lower() == "don't know":
                    results[model]["Don't Know"] += 1
                elif pred == correct:
                    results[model]["Correct"] += 1
                else:
                    results[model]["Incorrect"] += 1

# Build DataFrame
df = pd.DataFrame(results).T
df.index.name = 'Model'

# Print summary to console
print("\nModel Performance Summary:\n")
print(df.to_string())
print()

# Create combined grouped bar chart
x = np.arange(len(models))
width = 0.25
color_map = {'Correct': 'green', 'Incorrect': 'yellow', "Don't Know": 'red'}

fig, ax = plt.subplots(figsize=(8, 5))
for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in models]
    ax.bar(x + (i-1)*width, values, width, label=metric, color=color_map[metric])

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Count")
ax.set_xlabel("Model")
ax.set_title("Comparison of Model Performance Across Metrics")
ax.legend()

plt.tight_layout()

# Save to PNG in LeaderBoard folder
output_png = r"D:\Video_Analysis\LeaderBoard\combined_performance.png"
plt.savefig(output_png)
print(f"Saved combined chart to: {output_png}")
