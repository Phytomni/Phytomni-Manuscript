import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("score.tsv", sep="\t")
df = df.drop(["Gene", "Expert"], axis=1)

types = df["Species"].unique()
models = ["Claude", "Grok", "OpenAI", "Gemini", "Phytomni"]
ratings = ["R1", "R2", "R3", "R4", "R5"]

counts_data = []

for t in types:
    type_data = df[df["Species"] == t]
    for model in models:
        counts = type_data[model].value_counts().reindex(ratings, fill_value=0)
        for rating in ratings:
            counts_data.append({"Species": t, "Model": model, "Rating": rating, "Count": counts[rating]})

counts_df = pd.DataFrame(counts_data)

fig, axes = plt.subplots(5, 1, figsize=(8, 16))
axes = axes.flatten()

colors = ["#007cb9", "#63b3e4", "#9fccf2", "#cfe5f8", "#e6f2fa", "#eeeeee"]

for i, t in enumerate(types):
    ax = axes[i]
    type_data = counts_df[counts_df["Species"] == t]

    model_data = {}
    for model in models:
        model_ratings = type_data[type_data["Model"] == model]
        model_data[model] = model_ratings.set_index("Rating")["Count"].reindex(ratings, fill_value=0)

    stack_df = pd.DataFrame(model_data)

    bar_height = 0.6
    y_pos = np.arange(len(models))
    left = np.zeros(len(models))

    for j, rating in enumerate(ratings):
        ax.barh(y_pos, stack_df.loc[rating], bar_height, label=rating, left=left, color=colors[j])
        left += stack_df.loc[rating]

    ax.set_title(t)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("")
    ax.set_xticks([])

    for k, model in enumerate(models):
        left = 0
        for rating in ratings:
            value = stack_df.loc[rating, model]
            if value > 0:
                ax.text(left + value / 2, k, str(value), ha="center", va="center", fontweight="bold")
            left += value

fig.suptitle("Evaluation of Model Performance by Species", fontsize=16)

handles, labels = axes[0].get_legend_handles_labels()
ax.legend(title="Ratings", bbox_to_anchor=(1.2, 1), loc="right")
# fig.legend(handles, labels, title="Ratings", loc="")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("model_ratings_by_species.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
