
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
DATASET 1: GLOBAL NOISE
"""
# average_16 = "61.29% 9.22% 44.11% 62.35% 68.06% 31.86% 38.66% 66.53% 67.01% 49.26% 47.17% 66.74% 70.49% 42.98% 40.98% 62.35% 62.04% 45.83% 45.42% 60.72% 97.00% 98.24% 84.85% 97.46% 96.44% 94.23% 76.90% 96.56% 96.90% 93.98% 83.97% 97.14% 96.94% 87.00% 83.21% 96.84% 94.59% 78.78% 79.34% 95.99% 99.92% 99.85% 99.84% 99.83% 99.93% 99.89% 99.87% 99.88% 99.94% 99.90% 99.90% 99.90% 99.96% 99.94% 99.92% 99.89% 99.95% 99.86% 99.90% 99.85% 99.05% 99.32% 98.74% 99.10% 98.98% 99.52% 99.37% 99.26% 99.67% 99.50% 99.76% 99.68% 99.85% 86.50% 99.73% 99.84% 96.57% 67.79% 84.81% 93.14% 99.99% 99.97% 99.99% 99.99% 99.98% 99.96% 99.98% 99.98% 99.90% 99.19% 99.83% 99.89% 99.29% 71.26% 90.06% 99.32% 83.19% 41.62% 83.23% 73.59% 100.00% 99.99% 100.00% 100.00% 99.99% 90.37% 99.98% 99.98% 99.97% 88.48% 99.96% 99.97% 96.97% 88.76% 93.49% 99.48% 55.76% 23.74% 67.15% 68.09% 99.88% 99.87% 99.86% 99.87% 99.59% 99.50% 99.52% 99.61% 90.27% 98.42% 97.96% 97.73% 78.29% 43.69% 62.79% 78.11% 19.91% 7.25% 35.70% 22.38% 99.99% 99.98% 99.99% 99.99% 99.89% 99.84% 99.86% 99.75% 92.56% 71.74% 91.14% 93.39% 31.55% 16.33% 27.06% 32.04% 0.71% 1.43% 2.11% 0.76%"

# accuracies = [float(p.strip('%')) for p in average_16.split()]

# datasets = ["Church Outdoors", "Coffee Shop", "Conference Center", "Hamburger", 
#             "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]

# data = {
#     'Dataset': [d for d in datasets for _ in range(20)],
#     'Pruning Layer': (['None', '2', '6', '10'] * 5) * 8,
#     'Noise': [n*32 for d in range(8) for n in range(5) for _ in range(4)],
#     'Accuracy': accuracies
# }

# df = pd.DataFrame(data)

# # Set Styling
# sns.set_theme(style="whitegrid")
# sns.set_context("talk")

# # Graph 1: Detailed view per Dataset
# g = sns.FacetGrid(df, col="Dataset", col_wrap=4, hue="Pruning Layer", 
#                   height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
# g.add_legend(title="Pruning Layer")
# g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Average Performance Across Different Images", fontsize=24, fontweight='bold')
# plt.xticks(df['Noise'].unique().astype(int))
# plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# # GRAPH 2: Aggregated Average
# plt.figure(figsize=(8, 4))
# sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning Layer', 
#              style='Pruning Layer', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# plt.title("Average Performance Decay Under Noise", fontsize=20, fontweight='bold')
# plt.ylabel("Average Accuracy (%)")
# plt.xlabel("Gaussian Noise Level (SD)")
# plt.xticks(df['Noise'].unique().astype(int))
# plt.legend(title="Pruning Layer", loc='lower left')

# plt.tight_layout()
# plt.show()

"""
DATASET 2: LOCALIZED NOISE
"""

average_16 = "99.15% 99.16% 98.74% 96.59% 99.47% 97.61% 82.00% 87.49% 99.38% 99.14% 96.24% 92.41% 99.06% 99.09% 98.09% 95.35% 99.99% 99.90% 97.52% 78.73% 99.98% 99.90% 96.81% 76.33% 99.99% 99.92% 97.01% 87.03% 99.99% 99.90% 98.98% 80.49% 100.00% 99.97% 99.07% 95.71% 99.99% 81.87% 30.89% 0.45% 99.99% 99.88% 99.06% 83.98% 100.00% 99.98% 99.07% 87.10% 99.32% 88.26% 54.61% 28.16% 99.15% 90.17% 49.76% 26.11% 97.21% 62.50% 30.98% 9.78% 98.88% 87.28% 33.34% 20.68% 97.02% 79.38% 8.70% 2.49% 98.01% 26.12% 5.09% 1.01% 98.37% 47.75% 5.19% 1.12% 98.56% 37.69% 6.76% 0.63%"
accuracies = [float(p.strip('%')) for p in average_16.split()]
pruning = "None None None None 2 2 2 2 6 6 6 6 10 10 10 10"
datasets = ["Hamburger", "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]


data = {
    'Dataset': [d for d in datasets for _ in range(16)],
    'Pruning Layer': pruning.split()*5,
    'Noise': ([32, 64, 96, 128] * 4) * 5,
    'Accuracy': accuracies
}

df = pd.DataFrame(data)

# Set Styling
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Graph 1: Detailed view per Dataset
g = sns.FacetGrid(df, col="Dataset", col_wrap=3, hue="Pruning Layer", 
                  height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
g.add_legend(title="Pruning Layer")
g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Average Performance Across Different Images", fontsize=24, fontweight='bold')
plt.xticks(df['Noise'].unique().astype(int))
plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# GRAPH 2: Aggregated Average
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning Layer', 
             style='Pruning Layer', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
plt.title("Average Performance Decay Under Noise", fontsize=20, fontweight='bold')
plt.ylabel("Average Accuracy (%)")
plt.xlabel("Gaussian Noise Level (SD)")
plt.xticks(df['Noise'].unique().astype(int))
plt.legend(title="Pruning Layer", loc='lower left')

plt.tight_layout()
plt.show()