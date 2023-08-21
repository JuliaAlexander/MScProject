#Examining Precision of clips from whole gibbon range

#-----------------------------------
#1. Load input data
#setwd

#a) Loading my labelled clips
Labelled_1st <- read.csv("../data/B1_human_labels,_Labelling_validating_set_v5,_corrected_owl.csv")

#b) Loading metadata of selected clips
Metadata_1st <- read.csv("../data/B2_classifier_predictions,_Validating_2021_dataset,_shuffled_with_model_predictions.csv")
Model_MT <- Metadata_1st[,c(1,9:12,19,20,26)]

#-----------------------------------
#2. Getting word labels

#a) Extracting one hot labels
Labels_binary <- Labelled_1st[,c(2,4,5,6,7)]

#b) converting one hot to words label
Labels_name <- Labels_binary
Labels_name$label_background[Labels_name$label_background == 0] <-  "non-background"
Labels_name$label_background[Labels_name$label_background == 1] <-  "background"
Labels_name$label_gc[Labels_name$label_gc == 0] <-  "non-gc"
Labels_name$label_gc[Labels_name$label_gc == 1] <-  "gc"
Labels_name$label_mm[Labels_name$label_mm == 0] <-  "non-mm"
Labels_name$label_mm[Labels_name$label_mm == 1] <-  "mm"
Labels_name$label_sc[Labels_name$label_sc == 0] <-  "non-sc"
Labels_name$label_sc[Labels_name$label_sc == 1] <-  "sc"

##Extra
## #c) saving word label table
## write.csv(Labels_name, "../results/B3_Label_name_precision,_clips_1_to_200,_from_t3.csv", row.names = F)

#d) Combining model prediction and my labels
Validating_data <- data.frame(Labels_name, Model_MT)

#---------------------------------------------------------
#3. Calculating precision
library(tidyverse)

#---------
#a. finding number correct in a category
#i) background
bg_table <- table(which(Validating_data$label_background=="background") %in% 
            which(Validating_data$MT_background == "background" & Validating_data$MT_gc=="non-gc" & Validating_data$MT_mm=="non-mm" & Validating_data$MT_sc=="non-sc"))
background_correct <- as.numeric(bg_table)[2] #49

#ii) gc
gc_table <- table(which(Validating_data$label_gc=="gc") %in% 
                    which(Validating_data$MT_gc == "gc" & Validating_data$MT_background=="non-background" & Validating_data$MT_mm=="non-mm" & Validating_data$MT_sc=="non-sc"))
gc_correct <- as.numeric(gc_table)[2] #2

#iii) mm
mm_table <- table(which(Validating_data$label_mm=="mm") %in% 
                    which(Validating_data$MT_mm == "mm" & Validating_data$MT_background=="non-background" & Validating_data$MT_gc=="non-gc" & Validating_data$MT_sc=="non-sc"))
mm_correct <- as.numeric(mm_table)[2] #44

#iv) sc
sc_table <- table(which(Validating_data$label_sc=="sc") %in% 
                    which(Validating_data$MT_sc == "sc" & Validating_data$MT_background=="non-background" & Validating_data$MT_gc=="non-gc" & Validating_data$MT_mm=="non-mm" ))
sc_correct <- as.numeric(sc_table)[2] #12

#--------

#b. misclassified examining
#i) background misclassified
misclass_background <- Validating_data[which(Validating_data$MT_background == "background" & Validating_data$label_background=="non-background"),]
background_semi <- 0 #as NA #as all gibbon calls are in incorrect category
background_incorrect <- length(which(misclass_background$label_gc=="gc" | misclass_background$label_mm=="mm" | misclass_background$label_sc=="sc"))
  #1 incorrect ###1 mm and 1 sc
#49 correct

#ii) gc
#Examining gc misclassified
misclass_gc <- Validating_data[which(Validating_data$MT_gc == "gc" & Validating_data$label_gc=="non-gc"),]
gc_incorrect <- length(which(misclass_gc$label_background == "background" & misclass_gc$label_mm=="non-mm" & misclass_gc$label_sc=="non-sc"))
gc_semi <- length(which(misclass_gc$label_mm=="mm" | misclass_gc$label_sc=="sc"))
#2 correct, 1 other gibbon - 1 mm, 47 incorrect - background

#iii) mm
#Examining mm misclassified
misclass_mm <- Validating_data[which(Validating_data$MT_mm == "mm" & Validating_data$label_mm=="non-mm"),]
mm_incorrect <- length(which(misclass_mm$label_background == "background" & misclass_mm$label_gc=="non-gc" & misclass_mm$label_sc=="non-sc"))
mm_semi <- length(which(misclass_mm$label_gc=="gc" | misclass_mm$label_sc=="sc"))
#44 correct, 6 incorrect, 0 other gibbon

#iv) sc
#Examining sc misclassified
misclass_sc <- Validating_data[which(Validating_data$MT_sc == "sc" & Validating_data$label_sc=="non-sc"),]
sc_incorrect <- length(which(misclass_sc$label_background == "background" & misclass_sc$label_gc=="non-gc" & misclass_sc$label_mm=="non-mm"))
sc_semi <- length(which(misclass_sc$label_mm=="mm" | misclass_sc$label_gc=="gc")) #17 other gibbon
#12 correct, 17 other gibbon - 17 mm incl one which is gc too, 21 incorrect

#--------------------------------------------------------------------------------

#4. Making Precision proportion table

#a) adding raw values to a data frame
Raw_matches_table_short <- data.frame(Call_type=c("background","gc","mm","sc"),
                                      Correct=c(background_correct, gc_correct, mm_correct, sc_correct),
                                      Other_gibbon=c(background_semi, gc_semi, mm_semi, sc_semi),
                                      Incorrect=c(background_incorrect, gc_incorrect, mm_incorrect, sc_incorrect))
#    Call_type Correct Other_gibbon Incorrect
# 1 background      49            0         1
# 2         gc       2            1        47
# 3         mm      44            0         6
# 4         sc      12           17        21


#b) getting proportions
Clips_per_group = 50
Proportion_table_short <- cbind(Call_type = Raw_matches_table_short[,1],
                                Raw_matches_table_short[,c(2:4)]/Clips_per_group)
#    Call_type Correct Other_gibbon Incorrect
# 1 background    0.98            0      0.02
# 2         gc    0.04         0.02      0.94
# 3         mm    0.88         0.00      0.12
# 4         sc    0.24         0.34      0.42


#c) converting table from wide to long format
library(data.table)
Proportion_table <- melt(setDT(Proportion_table_short), id.vars = "Call_type", variable.name = "Labelled_type")
# Call_type Labelled_type value
# 1: background       Correct  0.98
# 2:         gc       Correct  0.04
# 3:         mm       Correct  0.88
# 4:         sc       Correct  0.24
# 5: background  Other_gibbon  0.00
# 6:         gc  Other_gibbon  0.02
# 7:         mm  Other_gibbon  0.00
# 8:         sc  Other_gibbon  0.34
# 9: background     Incorrect  0.02
# 10:         gc     Incorrect  0.94
# 11:         mm     Incorrect  0.12
# 12:         sc     Incorrect  0.42

#5. Making graph
##Other gibbon label
ggplot(Proportion_table, aes(x=Call_type, y=value, 
                             group = factor(Labelled_type, levels=c("Incorrect", "Other_gibbon", "Correct")) ))+
  geom_bar(stat="identity", position="stack", aes(fill=factor(Labelled_type)))+
  scale_fill_manual(labels=c("Correct", "Other gibbon", "Incorrect"),
                    values=c("darkgreen", "blue", "red"))+
  guides(fill=guide_legend(reverse=TRUE))+
  labs(x="Call type", y="Proportion of detections", fill="Label")+
  scale_y_continuous(breaks=seq(0,1,0.2))+
  theme_light()
ggsave("../results/B4_15dii.Proportion_validate_precision_clip_1_to_200_barplot,_corrected_bg_bar,_other_gibbon.png", width=6, height=6)




