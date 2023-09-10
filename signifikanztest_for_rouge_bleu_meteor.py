import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv and only take the first 100 rows
df_gpt2 = pd.read_csv('results/results_from_gpt2.csv').head(1000)
df_ft_gpt2 = pd.read_csv('results/results_from_fine-tuned-gpt2.csv').head(1000)
df_custom = pd.read_csv('results/results_from_custom_transformer.csv').head(1000)

# List of dataframes
dfs = [df_gpt2, df_ft_gpt2, df_custom]
names = ['GPT-2', 'Fine-tuned GPT-2', 'Custom Transformer']

# Check the statistics for ROUGE, BLEU and METEOR
for df, name in zip(dfs, names):
    print(f'Statistics for {name}:')
    for score in ['bleu_score', 'rouge_score', 'meteor_score']:
        print(f'{score}: Mean: {df[score].mean()}, Standard Deviation: {df[score].std()}')
    print()

# Perform t-test for each metric
for i in range(len(dfs)):
    for j in range(i+1, len(dfs)):
        print(f'T-test between {names[i]} and {names[j]}:')
        for score in ['bleu_score', 'rouge_score', 'meteor_score']:
            t_stat, p_value = ttest_rel(dfs[i][score], dfs[j][score])
            print(f'{score}: T-statistic: {t_stat}, P-value: {p_value}')
        print()

# Histogram for each score
fig, axs = plt.subplots(3, len(dfs), figsize=(15, 15))

# Create a figure and a set of subplots
for i, score in enumerate(['bleu_score', 'rouge_score', 'meteor_score']):
    bins = np.linspace(0, 1, 30)  # Creating bins for histogram
    for j, df in enumerate(dfs):
        # Plot histogram for each metric
        axs[i, j].hist(df[score], bins, alpha=0.5, label=names[j])

        # Adding legend
        axs[i, j].legend(loc='upper right')

        # Add title and labels
        axs[i, j].set_title(f'Histogram of {score} scores for {names[j]}')
        axs[i, j].set_xlabel('Score')
        axs[i, j].set_ylabel('Frequency')

# Show the plot
plt.tight_layout()
plt.show()
