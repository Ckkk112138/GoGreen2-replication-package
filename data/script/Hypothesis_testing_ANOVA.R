install.packages(c("readxl", "dplyr", "ez"))
library(readxl)
library(dplyr)
library(ez)

# Read in the data from Excel
bert_data <- read_excel("E:/VU&UvA_CS/GreenLab2023/Bert.xlsx")
distilled_bert_data <- read_excel("E:/VU&UvA_CS/GreenLab2023/DistilledBert.xlsx")
gpt2_data <- read_excel("E:/VU&UvA_CS/GreenLab2023/GPT2.xlsx")
distilled_gpt2_data <- read_excel("E:/VU&UvA_CS/GreenLab2023/DistilledGPT2.xlsx")

# Add model names to each dataframe
bert_data$Model <- "BERT"
distilled_bert_data$Model <- "Distilled-BERT"
gpt2_data$Model <- "GPT-2"
distilled_gpt2_data$Model <- "Distilled-GPT-2"

# Combine all the data into one dataframe
df <- bind_rows(bert_data, distilled_bert_data, gpt2_data, distilled_gpt2_data)
#df <- df[, -7]

# Clean column names for consistency
colnames(df) <- c("Run", "Avg_CPU_Utilization", "Avg_Memory", "Avg_Total_Energy", "Avg_Inference_Time", "Model")
# Repeated Measures ANOVA for Avg CPU Utilization
anova_utilization <- ezANOVA(data = df,
                             dv = Avg_CPU_Utilization,
                             wid = Run,
                             within = Model,
                             type = 3)
print(anova_utilization)

# Repeated Measures ANOVA for Avg Memory
anova_memory <- ezANOVA(data = df,
                        dv = Avg_Memory,
                        wid = Run,
                        within = Model,
                        type = 3)
print(anova_memory)

# Repeated Measures ANOVA for Avg Total Energy
anova_energy <- ezANOVA(data = df,
                        dv = Avg_Total_Energy,
                        wid = Run,
                        within = Model,
                        type = 3)
print(anova_energy)

# Repeated Measures ANOVA for Avg Inference Time
anova_time <- ezANOVA(data = df,
                      dv = Avg_Inference_Time,
                      wid = Run,
                      within = Model,
                      type = 3)
print(anova_time)
