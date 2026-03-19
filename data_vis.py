
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# """
# DATASET 1: GLOBAL NOISE
# """
# average_16 = "99.05% 99.32% 98.74% 99.10% 98.98% 99.52% 99.37% 99.26% 99.67% 99.50% 99.76% 99.68% 99.85% 86.50% 99.73% 99.84% 96.57% 67.79% 84.81% 93.14% 99.99% 99.97% 99.99% 99.99% 99.98% 99.96% 99.98% 99.98% 99.90% 99.19% 99.83% 99.89% 99.29% 71.26% 90.06% 99.32% 83.19% 41.62% 83.23% 73.59% 100.00% 99.99% 100.00% 100.00% 99.99% 90.37% 99.98% 99.98% 99.97% 88.48% 99.96% 99.97% 96.97% 88.76% 93.49% 99.48% 55.76% 23.74% 67.15% 68.09% 99.88% 99.87% 99.86% 99.87% 99.59% 99.50% 99.52% 99.61% 90.27% 98.42% 97.96% 97.73% 78.29% 43.69% 62.79% 78.11% 19.91% 7.25% 35.70% 22.38% 99.99% 99.98% 99.99% 99.99% 99.89% 99.84% 99.86% 99.75% 92.56% 71.74% 91.14% 93.39% 31.55% 16.33% 27.06% 32.04% 0.71% 1.43% 2.11% 0.76%"
# accuracies = [float(p.strip('%')) for p in average_16.split()]

# datasets = ["Hamburger", "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]

# data = {
#     'Dataset': [d for d in datasets for _ in range(20)],
#     'Pruning Layer': (['None', '2', '6', '10'] * 5) * 5,
#     'Noise': [n*32 for d in range(5) for n in range(5) for _ in range(4)],
#     'Accuracy': accuracies
# }

# df = pd.DataFrame(data)

# # Set Styling
# sns.set_theme(style="whitegrid")
# sns.set_context("talk")

# # Graph 1: Detailed view per Dataset
# g = sns.FacetGrid(df, col="Dataset", col_wrap=3, hue="Pruning Layer", 
#                   height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
# g.add_legend(title="Pruning Layer")
# g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
# g.fig.subplots_adjust(top=0.9)
# # g.fig.suptitle("Performance Of Different Images Under Global Noise (VIT-B/16)", fontsize=24, fontweight='bold')
# plt.xticks(df['Noise'].unique().astype(int))
# plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# # GRAPH 2: Aggregated Average
# plt.figure(figsize=(8, 4))
# sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning Layer', 
#              style='Pruning Layer', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# # plt.title("Average Performance Decay Under Global Noise (VIT-B/16)", fontsize=20, fontweight='bold')
# plt.ylabel("Average Accuracy (%)")
# plt.xlabel("Gaussian Noise Level (SD)")
# plt.xticks(df['Noise'].unique().astype(int))
# plt.legend(title="Pruning Layer", loc='lower left')

# plt.tight_layout()
# plt.show()

"""
DATASET 2: LOCALIZED NOISE
"""

average_16 = "99.15% 99.16% 98.74% 96.59% 99.47% 97.61% 82.00% 87.49% 99.38% 99.14% 96.24% 92.41% 99.06% 99.09% 98.09% 95.35% 99.99% 99.90% 97.95% 84.72% 99.98% 99.89% 94.49% 71.27% 99.99% 99.89% 98.68% 89.93% 99.99% 99.91% 98.76% 90.53% 100.00% 99.97% 99.07% 95.71% 99.99% 81.87% 30.89% 0.45% 99.99% 99.88% 99.06% 83.98% 100.00% 99.98% 99.07% 87.10% 99.32% 88.26% 54.61% 28.16% 99.15% 90.17% 49.76% 26.11% 97.21% 62.50% 30.98% 9.78% 98.88% 87.28% 33.34% 20.68% 97.02% 79.38% 8.70% 2.49% 98.01% 26.12% 5.09% 1.01% 98.37% 47.75% 5.19% 1.12% 98.56% 37.69% 6.76% 0.63%"
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
# g.fig.suptitle("Performance Of Different Images Under Localized Noise (VIT-B/16)", fontsize=24, fontweight='bold')
plt.xticks(df['Noise'].unique().astype(int))
plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# GRAPH 2: Aggregated Average
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning Layer', 
             style='Pruning Layer', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# plt.title("Average Performance Decay Under Localized Noise (VIT-B/16)", fontsize=20, fontweight='bold')
plt.ylabel("Average Accuracy (%)")
plt.xlabel("Gaussian Noise Level (SD)")
plt.xticks(df['Noise'].unique().astype(int))
plt.legend(title="Pruning Layer", loc='lower left')

plt.tight_layout()
plt.show()


# """
# DATASET 3: VIT-B/32 Global noise
# """
# average_32 = "99.00% 98.70% 98.86% 99.17% 98.96% 99.44% 99.40% 99.05% 99.28% 99.60% 99.76% 99.46% 99.87% 94.67% 99.88% 99.91% 99.19% 75.67% 99.26% 99.69% 100.00% 100.00% 100.00% 100.00% 100.00% 99.99% 100.00% 100.00% 99.95% 99.63% 99.84% 99.95% 93.63% 67.51% 95.32% 98.02% 49.46% 24.17% 65.90% 48.38% 100.00% 99.98% 99.99% 99.99% 99.98% 99.99% 99.95% 99.98% 98.24% 98.04% 98.50% 99.30% 70.37% 47.40% 63.58% 86.15% 20.97% 4.72% 24.41% 23.64% 99.98% 99.98% 99.95% 99.97% 99.95% 99.91% 99.85% 99.91% 90.10% 83.00% 84.07% 75.30% 34.48% 41.15% 56.58% 44.89% 30.38% 7.47% 24.30% 24.70% 99.98% 99.99% 99.99% 99.97% 99.83% 98.08% 99.77% 99.82% 88.84% 63.86% 89.55% 85.11% 6.21% 18.08% 7.30% 3.71% 0.10% 0.97% 1.37% 0.19%"
# accuracies = [float(p.strip('%')) for p in average_32.split()]

# datasets = ["Hamburger", "Ladybird", "Red Panda", "Scuba Diver", "Squirrel Monkey"]

# data = {
#     'Dataset': [d for d in datasets for _ in range(20)],
#     'Pruning Layer': (['None', '2', '6', '10'] * 5) * 5,
#     'Noise': [n*32 for d in range(5) for n in range(5) for _ in range(4)],
#     'Accuracy': accuracies
# }

# df = pd.DataFrame(data)

# # Set Styling
# sns.set_theme(style="whitegrid")
# sns.set_context("talk")

# # Graph 1: Detailed view per Dataset
# g = sns.FacetGrid(df, col="Dataset", col_wrap=3, hue="Pruning Layer", 
#                   height=4, aspect=1.5, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"])
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# g.map(sns.lineplot, "Noise", "Accuracy", marker="o", linewidth=2)
# g.add_legend(title="Pruning Layer")
# g.set_axis_labels("Noise Level (SD)", "Accuracy (%)")
# g.fig.subplots_adjust(top=0.9)
# # g.fig.suptitle("Performance Of Different Images Under Global Noise (VIT-B/32)", fontsize=24, fontweight='bold')
# plt.xticks(df['Noise'].unique().astype(int))
# plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

# # GRAPH 2: Aggregated Average
# plt.figure(figsize=(8, 4))
# sns.lineplot(data=df, x='Noise', y='Accuracy',  hue='Pruning Layer', 
#              style='Pruning Layer', markers=True, dashes=False, palette=["#3b0f70", "#8c2981","#db0a3e", "#fe9f6d"], linewidth=2, errorbar=None)
# # plt.title("Average Performance Decay Under Global Noise (VIT-B/32)", fontsize=20, fontweight='bold')
# plt.ylabel("Average Accuracy (%)")
# plt.xlabel("Gaussian Noise Level (SD)")
# plt.xticks(df['Noise'].unique().astype(int))
# plt.legend(title="Pruning Layer", loc='lower left')

# plt.tight_layout()
# plt.show()