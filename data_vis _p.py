
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
DATASET 1: GLOBAL NOISE
"""
average_16p = "99.15% 94.79% 92.64% 81.76% 62.05% 98.94% 72.47% 85.85% 61.83% 42.36% 92.11% 55.28% 18.14% 25.17% 7.76% 27.21% 4.05% 7.53% 3.79% 11.70% 54.20% 99.16% 98.34% 73.97% 26.70% 22.20% 99.87% 75.96% 11.57% 7.17% 48.27% 68.09% 32.34% 16.41% 0.56% 37.63% 34.80% 11.36% 0.47% 0.70% 100.00% 88.16% 74.17% 42.41% 18.21% 3.84% 48.86% 37.01% 13.52% 5.18% 25.03% 11.78% 2.23% 0.19% 1.08% 1.82% 1.38% 0.03% 0.03% 0.04% 99.85% 98.84% 94.20% 40.36% 7.93% 2.10% 97.62% 85.58% 39.01% 0.62% 1.35% 14.27% 4.89% 5.73% 9.46% 0.51% 1.81% 0.29% 0.42% 0.04% 99.95% 70.20% 67.28% 8.72% 1.25% 4.27% 41.37% 11.37% 0.71% 0.37% 2.54% 23.70% 0.11% 0.04% 0.06% 0.19% 1.23% 0.01% 0.00% 0.00%"
accuracies = [float(p.strip('%')) for p in average_16p.split()]

pruning = "78 78 78 78 78 58 58 58 58 58 38 38 38 38 38 18 18 18 18 18 78 78 78 78 78 58 58 58 58 58 38 38 38 38 38 18 18 18 18 18 78 78 78 78 78 58 58 58 58 58 38 38 38 38 38 18 18 18 18 18 78 78 78 78 78 58 58 58 58 58 38 38 38 38 38 18 18 18 18 18 78 78 78 78 78 58 58 58 58 58 38 38 38 38 38 18 18 18 18 18"
datasets = ["Hamburger", "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]

data = {
    'Dataset': [d for d in datasets for _ in range(20)],
    'Pruning amount': [int(x) for x in pruning.split()],
    'Noise': ([0, 32, 64, 96, 128] * 4) * 5,
    'Accuracy': accuracies
}

df = pd.DataFrame(data)

# Set Styling
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Graph 1: Detailed view per Dataset
g = sns.FacetGrid(df, col="Dataset", col_wrap=3, hue="Pruning amount", 
                  height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
g.add_legend(title="Tokens Left")
g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Performance Of Different Images Under Global Noise With Increased Pruning (VIT-B/16)", fontsize=40, fontweight='bold')
plt.xticks(df['Noise'].unique().astype(int))
plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# GRAPH 2: Aggregated Average
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning amount', 
             style='Pruning amount', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# plt.title("Average Performance Decay Under Localized Noise With Increased Pruning (VIT-B/16)", fontsize=40, fontweight='bold')
plt.ylabel("Average Accuracy (%)")
plt.xlabel("Gaussian Noise Level (SD)")
plt.xticks(df['Noise'].unique().astype(int))
plt.legend(title="Tokens Left", loc='lower left')

plt.tight_layout()
plt.show()
# exit()
"""
DATASET 2: LOCALIZED NOISE
"""
average_16 = "99.28% 90.22% 85.43% 85.60% 98.94% 88.88% 88.11% 89.04% 98.46% 86.50% 87.74% 86.54% 9.05% 8.44% 9.30% 10.87% 99.98% 99.86% 78.32% 17.17% 99.94% 99.27% 44.34% 5.54% 89.44% 56.72% 13.61% 3.00% 7.03% 5.16% 1.78% 3.48% 42.25% 63.47% 7.58% 0.17% 11.83% 1.05% 0.10% 0.04% 3.89% 0.06% 0.02% 0.02% 1.95% 0.03% 0.01% 0.01% 99.45% 91.35% 47.54% 20.11% 30.85% 39.51% 24.58% 4.75% 0.71% 0.16% 0.13% 0.05% 0.20% 0.03% 0.01% 0.02% 85.55% 10.96% 3.39% 0.48% 87.08% 6.00% 5.50% 1.18% 0.95% 8.97% 11.11% 2.41% 0.02% 0.01% 0.01% 0.01%"

accuracies = [float(p.strip('%')) for p in average_16.split()]

pruning = "78 78 78 78 58 58 58 58 38 38 38 38 18 18 18 18 78 78 78 78 58 58 58 58 38 38 38 38 18 18 18 18 78 78 78 78 58 58 58 58 38 38 38 38 18 18 18 18 78 78 78 78 58 58 58 58 38 38 38 38 18 18 18 18 78 78 78 78 58 58 58 58 38 38 38 38 18 18 18 18"

datasets = ["Hamburger", "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]

data = {
    'Dataset': [d for d in datasets for _ in range(16)],
    'Pruning amount': [int(x) for x in pruning.split()],
    'Noise': ([32, 64, 96, 128] * 4) * 5,
    'Accuracy': accuracies
}


df = pd.DataFrame(data)

# Set Styling
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Graph 1: Detailed view per Dataset
g = sns.FacetGrid(df, col="Dataset", col_wrap=3, hue="Pruning amount", 
                  height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
g.add_legend(title="Tokens Left")
g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Average Performance Decay Under Localized Noise With Increased Pruning (VIT-B/16)", fontsize=24, fontweight='bold')
plt.xticks(df['Noise'].unique().astype(int))
plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# GRAPH 2: Aggregated Average
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning amount', 
             style='Pruning amount', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# plt.title("Average Performance Decay Under Localized Noise With Increased Pruning (VIT-B/16)", fontsize=20, fontweight='bold')
plt.ylabel("Average Accuracy (%)")
plt.xlabel("Gaussian Noise Level (SD)")
plt.xticks(df['Noise'].unique().astype(int))
plt.legend(title="Tokens Left", loc='lower left')

plt.tight_layout()
plt.show()