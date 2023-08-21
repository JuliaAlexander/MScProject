##Getting 95% precision threshold

##adapting from getting P975 for 7CNNs script

#1. Libraries
import pandas as pd
from pathlib import Path
#import random
import numpy as np
from sklearn.metrics import precision_recall_curve

#2. Set seed for pytorch and python
#torch.manual_seed(0)
#random.seed(0)
np.random.seed(0)

#3. Loading gibbon data
gibbon_table = pd.read_csv(Path("J1_Clip_metadata_v2,_row_34_301_gc.csv"))
gibbon_table.head()
#    Clip_Number                               Clip_Path Location  Group_Name  Group_ID                                          Long_Path
# 0            1   label_groups/background/1_bird_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/1_bi...
# 1            2             label_groups/mm/2_mm_D1.WAV       D1          mm         2      ./Audio_Label_194/label_groups/mm/2_mm_D1.WAV
# 2            3   label_groups/background/3_bird_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/3_bi...
# 3            4             label_groups/mm/4_mm_D1.WAV       D1          mm         2      ./Audio_Label_194/label_groups/mm/4_mm_D1.WAV
# 4            5  label_groups/background/5_other_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/5_ot...
gibbon_table['filename'] = gibbon_table['Long_Path']
#making labels data frame
labels = pd.DataFrame(index=gibbon_table['filename'])
labels['background'] = [1 if l=='background' else 0 for l in gibbon_table['Group_Name']]
labels['gc'] = [1 if l=='gc' else 0 for l in gibbon_table['Group_Name']]
labels['mm'] = [1 if l=='mm' else 0 for l in gibbon_table['Group_Name']]
labels['sc'] = [1 if l=='sc' else 0 for l in gibbon_table['Group_Name']]
labels.head(3)
#                                                     background  gc  mm  sc
# filename                                                                  
# ./Audio_Label_194/label_groups/background/1_bir...           1   0   0   0
# ./Audio_Label_194/label_groups/mm/2_mm_D1.WAV                0   0   1   0
# ./Audio_Label_194/label_groups/background/3_bir...           1   0   0   0


################################################################################################################
#4. split by location
    # Validation Locations: [8, 5]
    # Testing Locations: [6, 4]
    #Test_clips = pd.DataFrame()
#a) Make Training clips subset
Train_clips = gibbon_table.loc[ (gibbon_table["Location"] != "D6") & (gibbon_table["Location"] != "D4") & 
                               (gibbon_table["Location"] != "D8") & (gibbon_table["Location"] != "D5") ]
Train_clips.shape #(7010, 7)
Train_clips.head()
#    Clip_Number                               Clip_Path Location  Group_Name  Group_ID                                          Long_Path
# 0            1   label_groups/background/1_bird_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/1_bi...
# 1            2             label_groups/mm/2_mm_D1.WAV       D1          mm         2      ./Audio_Label_194/label_groups/mm/2_mm_D1.WAV
# 2            3   label_groups/background/3_bird_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/3_bi...
# 3            4             label_groups/mm/4_mm_D1.WAV       D1          mm         2      ./Audio_Label_194/label_groups/mm/4_mm_D1.WAV
# 4            5  label_groups/background/5_other_D1.WAV       D1  background         0  ./Audio_Label_194/label_groups/background/5_ot...
#b) make Validation clips subset
Validation_clips = gibbon_table.loc[ (gibbon_table["Location"] == "D8") | (gibbon_table["Location"] == "D5") ]
Validation_clips.shape #(675, 7)
Validation_clips.head()
#       Clip_Number                                  Clip_Path Location  Group_Name  Group_ID                                          Long_Path
# 3810         3811  label_groups/background/3811_noise_D5.WAV       D5  background         0  ./Audio_Label_194/label_groups/background/3811...
# 3811         3812  label_groups/background/3812_noise_D5.WAV       D5  background         0  ./Audio_Label_194/label_groups/background/3812...
# 3812         3813  label_groups/background/3813_noise_D5.WAV       D5  background         0  ./Audio_Label_194/label_groups/background/3813...
# 3813         3814  label_groups/background/3814_noise_D5.WAV       D5  background         0  ./Audio_Label_194/label_groups/background/3814...
# 3814         3815  label_groups/background/3815_noise_D5.WAV       D5  background         0  ./Audio_Label_194/label_groups/background/3815..
#c) make Test clips subset
Test_clips = gibbon_table.loc[ (gibbon_table["Location"] == "D6") | (gibbon_table["Location"] == "D4") ]
Test_clips.shape #(1217, 7)
Test_clips.head()
#       Clip_Number                       Clip_Path Location Group_Name  Group_ID                                         Long_Path
# 3322         3323  label_groups/mm/3323_mm_D4.WAV       D4         mm         2  ./Audio_Label_194/label_groups/mm/3323_mm_D4.WAV
# 3323         3324  label_groups/mm/3324_mm_D4.WAV       D4         mm         2  ./Audio_Label_194/label_groups/mm/3324_mm_D4.WAV
# 3324         3325  label_groups/mm/3325_mm_D4.WAV       D4         mm         2  ./Audio_Label_194/label_groups/mm/3325_mm_D4.WAV
# 3325         3326  label_groups/mm/3326_mm_D4.WAV       D4         mm         2  ./Audio_Label_194/label_groups/mm/3326_mm_D4.WAV
# 3326         3327  label_groups/mm/3327_mm_D4.WAV       D4         mm         2  ./Audio_Label_194/label_groups/mm/3327_mm_D4.WAV
    #       Clip_Number  ...                                          filename
    # 3322         3323  ...  ./Audio_Label_194/label_groups/mm/3323_mm_D4.WAV
    # 3323         3324  ...  ./Audio_Label_194/label_groups/mm/3324_mm_D4.WAV
    # 3324         3325  ...  ./Audio_Label_194/label_groups/mm/3325_mm_D4.WAV
    # 3325         3326  ...  ./Audio_Label_194/label_groups/mm/3326_mm_D4.WAV
    # 3326         3327  ...  ./Audio_Label_194/label_groups/mm/3327_mm_D4.WAV
##7010+675+1217 #=8902
Train_Validation_clips = gibbon_table.loc[(gibbon_table["Location"] != "D6") & (gibbon_table["Location"] != "D4") ]
Train_Validation_clips.shape #(7685, 6)

#d) making one hot table function
def make_label_table(clips_table):
    """creates label table with each category as a column of 1s and 0s
    NB Input table needs to have index column with filenames"""
    label_table = pd.DataFrame(index=clips_table['filename'])
    label_table['background'] = [1 if l=='background' else 0 for l in clips_table['Group_Name']]
    label_table['gc'] = [1 if l=='gc' else 0 for l in clips_table['Group_Name']]
    label_table['mm'] = [1 if l=='mm' else 0 for l in clips_table['Group_Name']]
    label_table['sc'] = [1 if l=='sc' else 0 for l in clips_table['Group_Name']]
    return label_table
#e) Making one hot label subsets
#make_label_table(gibbon_table).head()
train_labels = make_label_table(Train_clips)
validation_labels = make_label_table(Validation_clips)
test_labels = make_label_table(Test_clips)
train_validation_labels = make_label_table(Train_Validation_clips)

#checking subset labels df
train_labels.shape #(7010, 4)
validation_labels.shape #(675, 4)
test_labels.shape # (1217, 4)
test_labels.head()
# Out[60]: 
#                                                   background  gc  mm  sc
# filename                                                                
# ./Audio_Label_194/label_groups/mm/3323_mm_D4.WAV           0   0   1   0
# ./Audio_Label_194/label_groups/mm/3324_mm_D4.WAV           0   0   1   0
# ./Audio_Label_194/label_groups/mm/3325_mm_D4.WAV           0   0   1   0
# ./Audio_Label_194/label_groups/mm/3326_mm_D4.WAV           0   0   1   0
# ./Audio_Label_194/label_groups/mm/3327_mm_D4.WAV           0   0   1   0


    # train_df = train_labels
    # validation_df = validation_labels
    # test_df = test_labels
    # train_df2 = train_labels.sample(frac = 1)

#f) shuffling rows in label dataset
train_df = train_labels.sample(frac = 1)
validation_df = validation_labels.sample(frac = 1)
test_df = test_labels.sample(frac = 1)
train_validation_df = train_validation_labels.sample(frac=1)

################################################################################################################

#5. Functions for label and single prediction columns
#a. function for adding label group column
def add_lab_col(test_predict_table, test_actual_labels):    
    #test_df is test_actual_labels
    #test_predict_label is test_predict_table
    test_predict_table["label"] = "NaN"
    for i in list(range(0,test_predict_table.shape[0])):
        if test_actual_labels.iloc[i,0] == 1: #background
            test_predict_table.iloc[i,5] = "background" #label column
        elif test_actual_labels.iloc[i,1] == 1: #gc
            test_predict_table.iloc[i,5] = "gc"
        elif test_actual_labels.iloc[i,2] == 1: #mm
            test_predict_table.iloc[i,5] = "mm"
        elif test_actual_labels.iloc[i,3] == 1: #sc
            test_predict_table.iloc[i,5] = "sc"
    return(test_predict_table)
            #  adding actual labels to V11 predict table
            # V11_predict_label = add_lab_col(V11_train_predict, train_df)
            # V11_predict_label.head()
            # #                                             filename  background         gc         mm         sc       label
            # # 0   ./Audio_Label_194/label_groups/sc/4898_sc_D9.WAV   -4.308016  -8.638646  -6.445943   4.039383          sc
            # # 1   ./Audio_Label_194/label_groups/sc/2636_sc_D3.WAV   -1.621667  -7.616260  -5.672320   1.395117          sc
            # # 2  ./Audio_Label_194/label_groups/background/2934...   15.808376 -19.131403 -14.031315 -17.670357  background
            # # 3   ./Audio_Label_194/label_groups/sc/2697_sc_D3.WAV   -8.346809 -11.992164  -9.372799   7.915843          sc
            # # 4  ./Audio_Label_194/label_groups/background/1894...    9.762295 -12.976413  -9.806648 -10.322445  background
#b. Adding single predicted category column
# Maximum category per row - predicted group
def PG_row_max(test_predict_label):
    """Adding predicted group column, 
    based on which of four category has largest predicted value"""
    test_pred_4groups = test_predict_label[['background', 'gc', 'mm', 'sc']].copy()
    max_true = test_pred_4groups.eq(test_pred_4groups.max(axis=1), axis=0)
    test_predict_label["predict_group"] = max_true.dot(max_true.columns)
    #test_predict_label.head()
    return(test_predict_label)
        # V11_predict_group = PG_row_max(V11_predict_label)
        # V11_predict_group.head()
        # #                                             filename  background         gc         mm         sc       label predict_group
        # # 0   ./Audio_Label_194/label_groups/sc/4898_sc_D9.WAV   -4.308016  -8.638646  -6.445943   4.039383          sc            sc
        # # 1   ./Audio_Label_194/label_groups/sc/2636_sc_D3.WAV   -1.621667  -7.616260  -5.672320   1.395117          sc            sc
        # # 2  ./Audio_Label_194/label_groups/background/2934...   15.808376 -19.131403 -14.031315 -17.670357  background    background
        # # 3   ./Audio_Label_194/label_groups/sc/2697_sc_D3.WAV   -8.346809 -11.992164  -9.372799   7.915843          sc            sc
        # # 4  ./Audio_Label_194/label_groups/background/1894...    9.762295 -12.976413  -9.806648 -10.322445  background    background



#6. Mutiple Threshold functions
##a. Separating label data into one vs rest
def binarise_category(prediction_table, category):
    category_labels = []
    for i in list(range(0,prediction_table.shape[0])): #1217 rows
        #raw label sc is called 1; rest is called 0
        if prediction_table.iloc[i]["label"] == category:
            category_labels.append(1)
        else:
            category_labels.append(0)
    return(category_labels)
            # background_labels = binarise_category(prediction_table=V11_predict_group, category="background")
            # background_labels[0:20]
            # # [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]

##b. Adding multiple prediction columns
# #Comparing clip labels and predictions
def add_MT_column(level, category, PR_table):
# def comparison_table2(level, category, PR_table):
    ##i) extracting predicted values
    predict_values = PR_table[[category]].copy()
    ##ii) prediction for mm category
    category_bool = predict_values >= level #gives True and  False
    ##iii) renaming True with mm and False with non-mm
    PR_group_PA = category_bool.copy() #present absent
    for i in list(range(0,predict_values.shape[0])):
        if category_bool.loc[i, category] ==True:
            PR_group_PA.loc[i, category] = category
        else:
            PR_group_PA.loc[i, category] = "non-"+category
    ##iv) Adding prediction column
    #R18_PR = R18_predict_group.copy()
    PR_table[["MT_"+category]]= PR_group_PA
    # R18_PR.loc[0:10,:]
    ##v) Comparison table for prediction
    comp_table = PR_table.groupby(['label', "MT_"+category ], as_index=False).size()
    return(comp_table)
                    # comparison_table2(level=background_level, category="background", PR_table=test_predict_group)
                    #         #         label   MT_background  size
                    #         # 0  background      background   443
                    #         # 1  background  non-background    40
                    #         # 2          gc  non-background    33
                    #         # 3          mm      background     5
                    #         # 4          mm  non-background   173
                    #         # 5          sc      background    10
                    #         # 6          sc  non-background   513
                    # #443/(443+5+10)*100 # = 96.72%


##c. Calculating precision, recall, f1 - Multi-threshold
    ##input is predict_table eg P90_test_predict_group
            ## Getting test_df raw scores
            # P90_test_predict_table = test_predict_table_raw.iloc[:,1:6].copy()
            # P90_test_predict_label = add_lab_col(P90_test_predict_table, test_df)
            # P90_test_predict_group = PG_row_max(P90_test_predict_label)
def get_MT_PRF(predict_table):
    #i) naming tables
    count_lab_pred = predict_table.groupby(['label','MT_gc','MT_mm','MT_sc','MT_background'], as_index=False).size()
                #          label   MT_gc   MT_mm   MT_sc   MT_background  size
                # 0   background      gc  non-mm  non-sc  non-background     1
                # 1   background  non-gc      mm  non-sc  non-background     1
                # 2   background  non-gc  non-mm  non-sc      background   444
                # 3   background  non-gc  non-mm  non-sc  non-background     2
    label_4g_total = predict_table.groupby(['label'], as_index=False).size()
                #         label  size
                # 0  background   483
                # 1          gc    33
                # 2          mm   178
                # 3          sc   523
    #ii) doing calculation - precision and recall
    for i in list(range(0, count_lab_pred.shape[0])): #over 19 rows
        for category in ["background","gc","mm","sc"]:
            if count_lab_pred.iloc[i]["label"] == count_lab_pred.iloc[i]["MT_"+category]: #for predict name being label
                ##recall - divides by label_category_total:
                P90_label_name_i = count_lab_pred.iloc[i]["label"]
                P90_label_table_row_number_i = label_4g_total.loc[label_4g_total["label"] == P90_label_name_i].index[0]
                P90_label_count_i = label_4g_total.loc[label_4g_total["label"] == P90_label_name_i]["size"][P90_label_table_row_number_i]
                count_lab_pred.at[i, "recall"] = (count_lab_pred.iloc[i]["size"]/P90_label_count_i)*100 ##recall                
                #precision - divides by predict_category_total:
                P90_category_test_predict_total = predict_table.groupby(["MT_"+category], as_index=False).size()
                P90_predict_table_row_number_i = P90_category_test_predict_total.loc[P90_category_test_predict_total["MT_"+category] == category].index[0]
                P90_predict_count_i = P90_category_test_predict_total.loc[P90_category_test_predict_total["MT_"+category] == category]["size"][P90_predict_table_row_number_i]
                count_lab_pred.at[i, "precision"] = (count_lab_pred.iloc[i]["size"]/P90_predict_count_i)*100 ##precision
    #iii) summarizing recall and precision
            #recall - over labels
            #precision - over predictions
    sum_PR = count_lab_pred.groupby(['label'], as_index=False).sum()
    summary_prf = sum_PR[["label", "recall", "precision"]].copy()
    #iv) finding f1 score
            #f1 score = (2*precision*recall)/(precision + recall)
    summary_prf["f1_score"] = "Empty"
    for j in list(range(0, summary_prf.shape[0])): #over 4 rows
        precision_i = summary_prf.loc[j, "recall"]
        recall_i = summary_prf.loc[j, "precision"]
        if (precision_i + recall_i != 0):
            f1_i = (2*precision_i*recall_i)/(precision_i + recall_i)
        else:
            f1_i = "NaN"
        summary_prf.at[j, "f1_score"] = f1_i
    # print(summary_prf)
    return(summary_prf)


##d. Finding precision threshold
def find_Pthreshold(category, Pfraction, predict_table):
    """Eg set Pfraction=0.95 for 95% precision threshold"""

    ##a) Getting label and prediction values for specified category
    y_label = binarise_category(prediction_table=predict_table, category=category)
    y_predict=predict_table[[category]]
    # background_labels = binarise_category(prediction_table=P95_TrainVal_predict_group, category="background")
    # background_predict_values = P95_TrainVal_predict_group[['background']].copy()

    ##b) Getting numpy arrays
    label_np = np.array(y_label)
    predict_np = y_predict.to_numpy()
    ##c) getting precision, recall, PR threshold values
    precision, recall, PR_thresholds = precision_recall_curve(label_np, predict_np)
    #d) calculating f1 score
    fscore = (2 * precision * recall) / (precision + recall)
    fscore_no_nan = fscore[np.isfinite(fscore)] #to remove any nan values, eg from division by 0

    ##e) Finding precision threshold
    P_index = abs(precision - Pfraction).argmin()
    ##P_index = abs(precision - Pfraction).nanargmin() #to skip nan values when finding min value
    print(category)
    print('Precision=%f, Recall=%f, index=%i' % (precision[P_index],recall[P_index], P_index))
    print('Best Threshold=%f, F-Score=%.3f' % (PR_thresholds[P_index], fscore[P_index]*100))
    return(PR_thresholds[P_index])

#===================================================================================================

#7. Function for extracting P95 thresholds 
#   #from loading TrainVal scores to saving csv of 97.5% precision thresholds

def get_P95_thresholds(scores_path, labels_df, thresholds_path):
    ##eg scores_path = "Train_Validation_df_vgg11.csv", and labels_df=train_validation_df  #Input path
    ##eg thresholds_path="Thresholds_95percent_precision_from_TrainVal_V11.csv"  #Output path
    #a) Loading train_val combined prediction scores
    scores_raw = pd.read_csv(scores_path).iloc[:,1:6]
    #b) Reordering rows
    scores_order = scores_raw.sort_values(by = "filename")
    labels_df_order = labels_df.sort_values(by="filename")
    #c) Adding label and single predict column
    table_predict_label = add_lab_col(scores_order, labels_df_order)
    table_predict_group = PG_row_max(table_predict_label)
    #d) Generating 95% precision threshold - multi-threshold on ordered rows
    Frac=0.95 #Setting precision level to generate threshold for
    background_level = find_Pthreshold(category="background", Pfraction=Frac, predict_table=table_predict_group)
    gc_level = find_Pthreshold(category="gc", Pfraction=Frac, predict_table=table_predict_group) 
    mm_level = find_Pthreshold(category="mm", Pfraction=Frac, predict_table=table_predict_group) 
    sc_level = find_Pthreshold(category="sc", Pfraction=Frac, predict_table=table_predict_group) 
    #e) Making dataframe with threshold values
    Levels_4gr = pd.DataFrame({'category':["background", "gc", "mm", "sc"], 'P5_threshold':[background_level, gc_level, mm_level, sc_level]})
        #      category  P95_threshold
        # 0  background       0.090411
        # 1          gc       1.664039
        # 2          mm       1.022020
        # 3          sc       0.383088
    #f) saving csv of threshold values
    Levels_4gr.to_csv(thresholds_path, index=False)
    #return(Levels_4gr)
    return(background_level, gc_level, mm_level, sc_level)


#8. Generating P95 thresholds
print("densenet121")
T4_D121 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_densenet121.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_densenet121.csv")

print("efficientnet_b0")
T4_Eb0 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_efficientnet_b0.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_efficientnet_b0.csv")
print("efficientnet_b4")
T4_Eb4 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_efficientnet_b4.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_efficientnet_b4.csv")

print("resnet_18")
T4_R18 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_resnet18.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_resnet18.csv")
print("resnet_50")
T4_R50 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_resnet50.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_resnet50.csv")
print("resnet_152")
T4_R152 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_resnet152.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_resnet152.csv")
print("vgg_11")
T4_V11 = get_P95_thresholds(scores_path="../VaTr_Test_subsets/VaTr_vgg11.csv", labels_df=train_validation_df, 
                    thresholds_path="../results/Thresholds_P95_vgg11.csv")



#=========================================================================================================================


#9. Apply 95% threshold to test_df

#a) Function for applying P95
def apply_P95(scores_path, labels_df, thresholds, model_name):
    ##eg thresholds=T4_D121
    #a) Loading train prediction scores
    scores_raw = pd.read_csv(scores_path).iloc[:,1:6]
    #b) Reordering rows
    scores_order = scores_raw.sort_values(by = "filename")
    labels_df_order = labels_df.sort_values(by="filename")
    #c) Adding label and single predict column
    table_predict_label = add_lab_col(scores_order, labels_df_order)
    table_predict_group = PG_row_max(table_predict_label)

    #d) Loading P95 thresholds
    background_level, gc_level, mm_level, sc_level = thresholds
    #e) Adding P95 columns
    add_MT_column(level=background_level, category="background", PR_table=table_predict_group)
    add_MT_column(level=gc_level, category="gc", PR_table=table_predict_group)
    add_MT_column(level=mm_level, category="mm", PR_table=table_predict_group)
    add_MT_column(level=sc_level, category="sc", PR_table=table_predict_group)
    #f) summary of precision recall f1 score - 97.5% Precision level onto test_df
    PRF_summary = get_MT_PRF(predict_table=table_predict_group)
    #97.5% Precision Multi-Threshold from TrainVal_df applied onto Test_df
        #         label     recall  precision   f1_score
        # 0  background  90.269151  97.539150  93.763441
        # 1          gc  15.151515  62.500000  24.390244
        # 2          mm  66.853933  93.700787  78.032787
        # 3          sc  86.042065  92.592593  89.197225
    PRF_summary[["model"]] = model_name
    print(PRF_summary)
    return(table_predict_group)

#b) examining thresholds on test df
print("densenet121")
PRF_D121 = apply_P95(scores_path="../VaTr_Test_subsets/Test_densenet121.csv", labels_df=test_df, 
                    thresholds=T4_D121, model_name="densenet121")
PRF_D121.to_csv("../results/Predictions_densenet121,_test_df_P95.csv", index=False)

print("efficientnet_b0")
PRF_Eb0 = apply_P95(scores_path="../VaTr_Test_subsets/Test_efficientnet_b0.csv", labels_df=test_df, 
                    thresholds=T4_Eb0, model_name="efficientnet_b0")
PRF_Eb0.to_csv("../results/Predictions_efficientnet_b0,_test_df_P95.csv", index=False)

print("efficientnet_b4")
PRF_Eb4 = apply_P95(scores_path="../VaTr_Test_subsets/Test_efficientnet_b4.csv", labels_df=test_df, 
                    thresholds=T4_Eb4, model_name="efficientnet_b4")
PRF_Eb4.to_csv("../results/Predictions_efficientnet_b4,_test_df_P95.csv", index=False)

print("resnet_18")
PRF_R18 = apply_P95(scores_path="../VaTr_Test_subsets/Test_resnet18.csv", labels_df=test_df, 
                    thresholds=T4_R18, model_name="resnet_18")
PRF_R18.to_csv("../results/Predictions_resnet18,_test_df_P95.csv", index=False)

print("resnet_50")
PRF_R50 = apply_P95(scores_path="../VaTr_Test_subsets/Test_resnet50.csv", labels_df=test_df, 
                    thresholds=T4_R50, model_name="resnet_50")
PRF_R50.to_csv("../results/Predictions_resnet50,_test_df_P95.csv", index=False)

print("resnet_152")
PRF_R152 = apply_P95(scores_path="../VaTr_Test_subsets/Test_resnet152.csv", labels_df=test_df, 
                    thresholds=T4_R152, model_name="resnet_152")
PRF_R152.to_csv("../results/Predictions_resnet152,_test_df_P95.csv", index=False)
      
print("vgg_11")
PRF_V11 = apply_P95(scores_path="../VaTr_Test_subsets/Test_vgg11.csv", labels_df=test_df, 
                    thresholds=T4_D121, model_name="vgg_11")
PRF_V11.to_csv("../results/Predictions_vgg11,_test_df_P95.csv", index=False)

