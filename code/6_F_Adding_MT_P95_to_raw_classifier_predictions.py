#Generating Multi-Threshold Values
    #Precision 95% Threshold VGG-11
    #using updated gc threshold value

#1. Libraries
import pandas as pd
from glob import glob

            # ##Function for adding MT columns
            # def file_add_MT(thresholds_path, )

#2. Loading 95% precision VGG11 thresholds
P95_Levels_4gr = pd.read_csv("Thresholds_P95_vgg11_t2.csv")

#3. Adding Multi-threshold Columns Function
def add_MT(levels, predict_table):
    #For each category
    for i in range(0,levels.shape[0]):
        #i) Extracting category name and category 95% precision threshold
        category_i = levels.iloc[i]["category"]
        P95_level_i = levels.iloc[i]["P95_threshold"]
        #ii) Evaluating whether clip is above or equal to threshold
        predict_values_i = predict_table[[category_i]]
        bool_category_i = predict_values_i >= P95_level_i
        #iii) Renaming True with eg mm and False with eg non-mm
        PA_group_i = bool_category_i.copy(deep=True) #present absent
        for i in list(range(0,predict_values_i.shape[0])):
            if bool_category_i.loc[i, category_i] ==True:
                PA_group_i.loc[i, category_i] = category_i
            else:
                PA_group_i.loc[i, category_i] = "non-"+category_i        
        ##iv) Adding prediction column
        predict_table[["MT_"+category_i]]= PA_group_i
    return(predict_table)

#4. Loading 2021 data raw predictions
raw_scores = glob("../results/raw_tables/*/*.csv")

#5. Generating 2021 data csv files with multi-thresholds
for file in raw_scores:
    MT_scores_i = add_MT(levels=P95_Levels_4gr, predict_table = pd.read_csv(file))
    name_i = file.split('/')[-1]
    MT_scores_i.to_csv("../results/MT_P95/MT_"+name_i)
