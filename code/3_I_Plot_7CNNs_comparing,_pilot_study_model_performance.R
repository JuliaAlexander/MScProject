#Summary 7CNNs multi threshold
 #Comparing Model Performance of 7 different convolutional neural network architectures

#P95 - 95% precision threshold level

#1. Libraries
library(data.table)
library(tidyverse)

#2. Load data
#setwd
P95_7CNNs <- read.table("I1_P95_7CNN_summary_v2.txt", header=T, row.names=NULL)[,c(2:6)]
	#Removed underscore in CNN names

#3. Line graph - P95 - multi threshold
  #plotting score vs category, coloured by model
# Plotting f1 score
ggplot(P95_table, aes(x=label,y=f1_score, col=model))+
  geom_point(size=4)+
  geom_line(aes(group=model), lwd=1.5)+
  theme_bw()+
  labs(x="Audio category", y="F1 score", col="Model")
ggsave("I2_7.P95_linegraph,_f1_score_vs_category,_by_7CNNs.png", width=9, height = 6)
# Warning message:
#   Removed 1 rows containing missing values (`geom_point()`). #as D121 has NAN value


