import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# %% plot GeneTuring
data = pd.read_excel("data.xlsx", sheet_name="GeneTuring")
data.set_index("Models", inplace=True)

colors = ["#ffffff", "#8287C1"]
gradient_cmap = LinearSegmentedColormap.from_list("gradient", colors)

plt.figure(figsize=(8, 8))
sns.heatmap(data, annot=True, cmap=gradient_cmap, vmin=0, vmax=1)
plt.title("Accuracy Assessment of GeneTuring")
plt.xticks(rotation=45, ha="right")
# plt.tick_params(axis="x", which="major", pad=5)
plt.xlabel("")
plt.ylabel("")

# plt.show()
plt.tight_layout()
plt.savefig("GeneTuring.pdf")


# %% plot pangu model
data = pd.read_excel("data.xlsx", sheet_name="pangu")
heatmap_data = data.pivot(index="Classification", columns="Model", values="Accuracy")

# colors = ["#00032C", "#8A0E80", "#CE0258", "#FBF703"]
colors = ["#ffffff", "#67AFD7"]
gradient_cmap = LinearSegmentedColormap.from_list("gradient", colors)


plt.figure(figsize=(8, 8))
sns.heatmap(heatmap_data, annot=True, cmap=gradient_cmap, vmin=0, vmax=1)
plt.title("Performance of Pangu Models on BI-1000")
plt.xticks(rotation=45, ha="right")
# plt.tick_params(axis="x", which="major", pad=5)
plt.xlabel("")
plt.ylabel("")

# plt.show()
plt.tight_layout()
plt.savefig("pangu.pdf")


# %% plot other models
data = pd.read_excel("data.xlsx", sheet_name="other")

left_color = "#ff9e04"
right_color = "#3179f0"

fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(
    data["model"],
    [-x for x in data["BI-1000"]],
    color=left_color,
    label="BI-1000",
)
ax.barh(
    data["model"],
    data["BI-65"],
    color=right_color,
    label="BI-65",
)
ax.axvline(0, color="#6e6e6e", linewidth=0.8)


# set x-axis limits and ticks
ax.set_xlim(-1.08, 1.08)
ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticklabels([1, 0.8, 0.6, 0.4, 0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# data labels
offset = 0.01
for i, value in enumerate(data["BI-1000"]):
    ax.text(-value - offset, i, f"{value:.2f}", ha="right", va="center", color=left_color)

for i, value in enumerate(data["BI-65"]):
    ax.text(value + offset, i, f"{value:.2f}", ha="left", va="center", color=right_color)

# add labels and title
ax.set_xlabel("Accuracy Rate")
ax.set_title("Performance of Comparative Models")
ax.legend()


# plt.show()
plt.tight_layout()
plt.savefig("other_models.pdf")

# %%
