import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

folder_bert = "/Users/eloise/Downloads/Data/bert_data/"
df_bert_1 = pd.read_csv(folder_bert + 'bert_runner_experiment_1/run_table.csv')
df_bert_2 = pd.read_csv(folder_bert + 'bert_runner_experiment_2/run_table.csv')
df_bert = pd.concat([df_bert_1, df_bert_2], ignore_index=True)
old_total_energy = df_bert.loc[5].total_energy
new_total_energy = old_total_energy - (-262099.31250) + 43.76563
df_bert.loc[5, "total_energy"] = new_total_energy
df_bert['avg_cpu'] = df_bert['avg_cpu'] * 100

print(old_total_energy)
print(new_total_energy)
print(df_bert)



folder_distilledbert = "/Users/eloise/Downloads/Data/distilledbert/"
df_distilledbert_1 = pd.read_csv(folder_distilledbert + 'distilledbert_runner_experiment_1/run_table.csv')
df_distilledbert_2 = pd.read_csv(folder_distilledbert + 'distilledbert_runner_experiment_2/run_table.csv')
df_distilledbert = pd.concat([df_distilledbert_1, df_distilledbert_2], ignore_index=True)
df_distilledbert['avg_cpu'] = df_distilledbert['avg_cpu'] * 100
print(df_distilledbert)

folder_gpt = "/Users/eloise/Downloads/Data/gpt_runner_experiment_1/"
df_gpt = pd.read_csv(folder_gpt + 'run_table.csv')
old_total_energy = df_gpt.loc[7].total_energy
new_total_energy = old_total_energy - (-262096.20313) + 46.21875
df_gpt.loc[7, "total_energy"] = new_total_energy
df_gpt['avg_cpu'] = df_gpt['avg_cpu'] * 100
print(df_gpt)

folder_distilledgpt = "/Users/eloise/Downloads/Data/distilledgpt_runner_experiment_1/"
df_distilledgpt = pd.read_csv(folder_distilledgpt + 'run_table.csv')
df_distilledgpt['avg_cpu'] = df_distilledgpt['avg_cpu'] * 100
print(df_distilledgpt)

print(df_bert["total_energy"].mean())
print(df_distilledbert["total_energy"].mean())
print(df_gpt["total_energy"].mean())
print(df_distilledgpt["total_energy"].mean())

energy_dfs = pd.DataFrame({'BERT': df_bert['total_energy'],
                            'DistilBERT': df_distilledbert['total_energy'],
                            'GPT2': df_gpt['total_energy'],
                            'DistilGPT2': df_distilledgpt['total_energy']})

plt.figure(figsize=(6, 8))  # Adjust the figure size if needed
sns.boxplot(data = energy_dfs, notch=True, showcaps=False, palette = "Set2")
plt.savefig('Energy.png')

df_bert['Model'] = 'BERT'
df_distilledbert['Model'] = 'DistilBERT'
df_gpt['Model'] = 'GPT2'
df_distilledgpt['Model'] = 'DistilGPT2'
combined_df = pd.concat([df_bert, df_distilledbert, df_gpt, df_distilledgpt], ignore_index=True)
combined_melted = pd.melt(combined_df, id_vars=['Model'], 
                          value_vars=['avg_cpu', 'avg_memory', 'total_inference_time'],
                          var_name='Metric', value_name='Value')

# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Plot for left y-axis (avg_cpu, avg_memory, total_energy)
# sns.boxplot(x='Model', y='Value', hue='Metric', 
#             data=combined_melted[combined_melted['Metric'] != 'total_inference_time'],
#             ax=ax1, width=0.6)

# # Add titles and labels for the left y-axis
# ax1.set_title('Model Comparison')
# ax1.set_xlabel('Model')
# ax1.set_ylabel('Value')

# # Create a second y-axis for total_inference_time
# ax2 = ax1.twinx()
# sns.boxplot(x='Model', y='Value', hue='Metric', 
#             data=combined_melted[combined_melted['Metric'] == 'total_inference_time'],
#             ax=ax2, width=0.3)

# # Add labels for the right y-axis
# ax2.set_ylabel('Total Inference Time')


fig, ax1 = plt.subplots(figsize=(10, 6))
models = combined_melted['Model'].unique()
# Create a color palette for the box plots
palette = sns.color_palette("hls", n_colors=3)

# Initialize x-position for each model
x_pos = range(1, len(models) * 2, 2)

width=0.6
# Plot for left y-axis (avg_cpu, avg_memory, total_energy)
for i, metric in enumerate(['avg_cpu', 'avg_memory']):
    values = [combined_melted[(combined_melted['Model'] == model) & (combined_melted['Metric'] == metric)]['Value'].tolist() for model in models]
    ax1.boxplot(values, positions=[pos - width + i * width for pos in x_pos], widths=width, labels=models, patch_artist=True, boxprops=dict(facecolor=palette[i]))

# Add titles and labels for the left y-axis
ax1.set_title('Model Comparison')
ax1.set_xlabel('Model')
ax1.set_ylabel('CPU/Memory Percentage')

# Create a second y-axis for total_inference_time
ax2 = ax1.twinx()
inference_time_values = [combined_melted[(combined_melted['Model'] == model) & (combined_melted['Metric'] == 'total_inference_time')]['Value'].tolist() for model in models]
ax2.boxplot(inference_time_values, positions=[pos + width for pos in x_pos], widths=width, labels=models, patch_artist=True, boxprops=dict(facecolor=palette[2]))

# Add labels for the right y-axis
ax2.set_ylabel('Total Inference Time')

# Set x-axis ticks and labels
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models)

# Define custom legends
legend_labels = ['avg_cpu', 'avg_memory', 'total_inference_time']
colors = palette
# Create a custom legend
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(3)]
# Add the legend to the figure
ax1.legend(custom_legend, legend_labels, loc='upper right')

# Adjust layout
plt.tight_layout()

plt.savefig('All.png')