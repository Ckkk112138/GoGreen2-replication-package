# Load necessary libraries
install.packages("readxl")
install.packages("car")
library(readxl)
library(car)

# Load the data
distilledGPT2 <- read_excel("E:/VU&UvA_CS/GreenLab2023/data_DistilledGPT2_run.xlsx")
GPT2 <- read_excel("E:/VU&UvA_CS/GreenLab2023/data_GPT2_run.xlsx")
# GPT2 <- GPT2[-c(560:797), ]
# Levene's Test for CPU utilization
leveneUtilization <- leveneTest(distilledGPT2$`CPU Utilization`, GPT2$`CPU Utilization`)
print(leveneUtilization)

# Levene's Test for CPU power
levenePower <- leveneTest(distilledGPT2$`CPU Power`, GPT2$`CPU Power`)
print(levenePower)

# Independent t-test for CPU utilization
ttestUtilization <- t.test(distilledGPT2$`CPU Utilization`, GPT2$`CPU Utilization`, var.equal = (leveneUtilization$`Pr(>F)`[1] > 0.05))
print(ttestUtilization)

# Independent t-test for CPU power
ttestPower <- t.test(distilledGPT2$`CPU Power`, GPT2$`CPU Power`)
print(ttestPower)
