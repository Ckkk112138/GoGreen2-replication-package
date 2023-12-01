import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

folder_bert = "/Users/eloise/Downloads/Data/bert_data/"
df_bert_1 = pd.read_csv(folder_bert + 'bert_runner_experiment_1/run_table.csv')
df_bert_2 = pd.read_csv(folder_bert + 'bert_runner_experiment_2/run_table.csv')
df_bert_3 = pd.read_csv(folder_bert + 'bert_runner_experiment_3/run_table.csv')
df_bert = pd.concat([df_bert_1, df_bert_2, df_bert_3], ignore_index=True)
df_bert = df_bert.drop([5]).reset_index()
df_bert['avg_cpu'] = df_bert['avg_cpu'] * 100
print(df_bert)

folder_distilledbert = "/Users/eloise/Downloads/Data/distilledbert/"
df_distilledbert_1 = pd.read_csv(folder_distilledbert + 'distilledbert_runner_experiment_1/run_table.csv')
df_distilledbert_2 = pd.read_csv(folder_distilledbert + 'distilledbert_runner_experiment_2/run_table.csv')
df_distilledbert = pd.concat([df_distilledbert_1, df_distilledbert_2], ignore_index=True)
df_distilledbert['avg_cpu'] = df_distilledbert['avg_cpu'] * 100
print(df_distilledbert)

folder_gpt = "/Users/eloise/Downloads/Data/gpt_runner_experiment_1/"
df_gpt_1 = pd.read_csv(folder_gpt + 'run_table.csv')
df_gpt_2 = pd.read_csv(folder_gpt + 'gpt_runner_experiment_2/run_table.csv')
df_gpt = pd.concat([df_gpt_1, df_gpt_2], ignore_index=True)
df_gpt = df_gpt.drop([7]).reset_index()
df_gpt['avg_cpu'] = df_gpt['avg_cpu'] * 100
print(df_gpt)

folder_distilledgpt = "/Users/eloise/Downloads/Data/distilledgpt_runner_experiment_1/"
df_distilledgpt = pd.read_csv(folder_distilledgpt + 'run_table.csv')
df_distilledgpt['avg_cpu'] = df_distilledgpt['avg_cpu'] * 100
print(df_distilledgpt)

print(df_bert["avg_cpu"].mean())
print(df_distilledbert["avg_cpu"].mean())
print(df_gpt["avg_cpu"].mean())
print(df_distilledgpt["avg_cpu"].mean())

# energy_dfs = pd.DataFrame({'BERT': df_bert['total_energy'],
#                         'DistilBERT': df_distilledbert['total_energy'],
#                         'GPT2': df_gpt['total_energy'],
#                         'DistilGPT2': df_distilledgpt['total_energy']})

# cpu_dfs = pd.DataFrame({'BERT': df_bert['avg_cpu'],
#                         'DistilBERT': df_distilledbert['avg_cpu'],
#                         'GPT2': df_gpt['avg_cpu'],
#                         'DistilGPT2': df_distilledgpt['avg_cpu']})

# memory_dfs = pd.DataFrame({'BERT': df_bert['avg_memory'],
#                         'DistilBERT': df_distilledbert['avg_memory'],
#                         'GPT2': df_gpt['avg_memory'],
#                         'DistilGPT2': df_distilledgpt['avg_memory']})

# time_dfs = pd.DataFrame({'BERT': df_bert['total_inference_time'],
#                             'DistilBERT': df_distilledbert['total_inference_time'],
#                             'GPT2': df_gpt['total_inference_time'],
#                             'DistilGPT2': df_distilledgpt['total_inference_time']})

# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.2)

# # Create a figure with four subplots
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# # Define the dataframes and titles
# dfs = [cpu_dfs, memory_dfs, energy_dfs, time_dfs]
# titles = ['Average CPU Usage (%)', 'Average Memory Usage (%)', 'Total Energy (J)',  'Total Inference Time (s)']

# # Loop through each subplot
# for i, ax in enumerate(axes.flat):
#     df = dfs[i]
#     title = titles[i]
    
#     # Create the box plot
#     sns.boxplot(data=df, palette="Set2", ax=ax)
    
#     # Set title
#     ax.set_title(title, fontsize=14)
    
#     # Set x and y labels
#     ax.set_xlabel("Models", fontsize=12)
#     ax.set_ylabel(title, fontsize=12)

# plt.tight_layout()
# plt.savefig('boxplot.pdf', format='pdf')