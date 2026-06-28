import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("data.xlsx")
long_df = df.melt(id_vars="species", var_name="model", value_name="result")
long_df["result"] = long_df["result"].astype(str).str.upper() == "TRUE"

acc_df = long_df.groupby(["species", "model"], as_index=False).agg(accuracy=("result", "mean"), n=("result", "size"))

species_counts = df["species"].value_counts()
species_order = species_counts.index.tolist()
species_order = [species_order[5]] + species_order[:5] + species_order[6:]

acc_df["species"] = pd.Categorical(acc_df["species"], categories=species_order, ordered=True)
model_order = ["Phyto-Reasoner", "Phyto-Chatbot", "GPT-5", "o3", "Gemini-2.5-Pro", "Claude-Opus-4.1", "Grok-3-Beta", "DeepSeek-V3", "DeepSeek-R1"]
acc_df["model"] = pd.Categorical(acc_df["model"], categories=model_order, ordered=True)
acc_df = acc_df.sort_values(["species", "model"])

# print(acc_df)
# acc_df.to_excel("model_accuracy_by_species.xlsx", index=False)

species_labels = [f"{sp} ({species_counts[sp]})" for sp in species_order]
custom_colors = {
    "Phyto-Reasoner": "#1e72b2",
    "Phyto-Chatbot": "#58A4C3",
    "GPT-5": "#F2C6DE",
    "o3": "#F29C9A",
    "Gemini-2.5-Pro": "#E59E65",
    "Claude-Opus-4.1": "#D6BA9E",
    "Grok-3-Beta": "#9C9A9C",
    "DeepSeek-V3": "#4D9489",
    "DeepSeek-R1": "#96CDA0",
}

species_per_row = 12
n_species = len(species_order)
n_rows = math.ceil(n_species / species_per_row)

species_chunks = [species_order[i : i + species_per_row] for i in range(0, n_species, species_per_row)]
fig, axes = plt.subplots(n_rows, 1, figsize=(18, 4 * n_rows), sharey=True)

if n_rows == 1:
    axes = [axes]

for ax, chunk in zip(axes, species_chunks):
    sub_df = acc_df[acc_df["species"].isin(chunk)].copy()

    sns.barplot(data=sub_df, x="species", y="accuracy", hue="model", order=chunk, ax=ax, palette=custom_colors)

    labels = [f"{sp} ({species_counts[sp]})" for sp in chunk]

    ax.set_xticks(range(len(chunk)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")

    if ax.get_legend() is not None:
        ax.get_legend().remove()

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(labels), 4))

fig.supxlabel("Species (n)")
fig.supylabel("Accuracy")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Uncomment to write the figure (output lands beside this script):
# plt.savefig("model_accuracy_by_species.pdf", format="pdf", bbox_inches="tight")
plt.show()
