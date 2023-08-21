# Applying pytorch tutorial
        #Using 5 second audio


##1. In terminal: 
# conda activate opensoundscape
# ipython3

#2. Libraries
from opensoundscape import CNN
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#3. Set seed for pytorch and python
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#4. Generate one-hot encoded labels
    #column with relative path to file
    #column with each class name 1=present, 0=absent
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

# #5. Split into training and validation sets
#     #randomly divide labelled samples
# train_df, validation_test_df = train_test_split(labels, test_size=0.2, random_state=1) #random_state arg for reproduible results
# validation_df, test_df = train_test_split(validation_test_df, test_size=0.5,random_state=1 )

################################################################################################################
#5. split by location
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


################################################################################################################
#Training on 5 seconds audio length

#6. Loading pretrained CNN
classes = train_df.columns # Index(['background', 'gc', 'mm', 'sc'], dtype='object')
	#using frozen feature extraction part:
	###import opensoundscape.ml.cnn_architectures as cnn_arch
	#from opensoundscape.ml import cnn_architectures
	#R18_arch = cnn_architectures.resnet18(num_classes=4, freeze_feature_extractor=True)
	#Eb0_arch = cnn_architectures.efficientnet_b0(num_classes=4, freeze_feature_extractor=True)
	#model_resnet18 = CNN(R18_arch, classes=classes, sample_duration=2.0)
	#model_EN_b0 = CNN(Eb0_arch, classes=classes, sample_duration=2.0)
model_resnet18 = CNN('resnet18', classes=classes, sample_duration=5.0)
#model_EN_b0 = CNN('efficientnet_b0', classes=classes, sample_duration=5.0)
model_vgg11_bn = CNN('vgg11_bn', classes=classes, sample_duration=5.0)
#model_EN_b4 = CNN('efficientnet_b4', classes=classes, sample_duration=5.0)
model_densenet121 = CNN('densenet121', classes=classes, sample_duration=5.0)
# Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /home/julia/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30.8M/30.8M [00:02<00:00, 11.8MB/s]
# model_resnet152 = CNN('resnet152', classes=classes, sample_duration=5.0)
model_resnet50 = CNN('resnet50', classes=classes, sample_duration=5.0)
# Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /home/julia/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:09<00:00, 10.5MB/s]
model_resnet152 = CNN('resnet152', classes=classes, sample_duration=5.0)
# Downloading: "https://download.pytorch.org/models/resnet152-f82ba261.pth" to /home/julia/.cache/torch/hub/checkpoints/resnet152-f82ba261.pth
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230M/230M [00:23<00:00, 10.4MB/s]


#7. Enabling 16000Hz sample rate
    #doing bypass so runs on 16000Hz sample rate audio
model_resnet18.preprocessor.pipeline.bandpass.bypass=True
#model_EN_b0.preprocessor.pipeline.bandpass.bypass=True
model_vgg11_bn.preprocessor.pipeline.bandpass.bypass=True
#model_EN_b4.preprocessor.pipeline.bandpass.bypass=True
model_densenet121.preprocessor.pipeline.bandpass.bypass=True
model_resnet50.preprocessor.pipeline.bandpass.bypass=True
model_resnet152.preprocessor.pipeline.bandpass.bypass=True


#8. Training CNN model
    # they recommend eg tens of epochs on eg hundreds of training files

#Resnet18
print("resnet 18")
model_resnet18.train(
    train_df = train_df,
    validation_df = validation_df,
    save_path='../saved_models/t8c_150gc/resnet18_B32/', #trained model will be saved here
    # save_path='./5_pytorch_R18_b0/saved_models/resnet18/', #trained model will be saved here
    epochs=30,
    #batch_size=8,
    batch_size=32,
    save_interval=1, #save model every 5 epochs, and best model always saved in addition
    num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
)
# ##EfficientNet_b0
# print("efficient net b0")
# model_EN_b0.train(
#     train_df = train_df,
#     validation_df = validation_df,
#     save_path='../saved_models/efficientnet_b0_location_150gc_B32/', #trained model will be saved here
#     # save_path='./5_pytorch_R18_b0/saved_models/efficientnet_b0_model/', #trained model will be saved here
#     epochs=20,
#     batch_size=32,
#     save_interval=1, #save model every 5 epochs, and best model always saved in addition
#     num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
# )
# #EfficientNet_b4
# print("efficient net b4")
# model_EN_b4.train(
#     train_df = train_df,
#     validation_df = validation_df,
#     save_path='../saved_models/efficientnet_b4_location_150gc_B32/', #trained model will be saved here
#     # save_path='./model/efficientnet_b4/', #trained model will be saved here
#     # save_path='./binary_train/', #trained model will be saved here
#     epochs=32,
#     batch_size=32,
#     save_interval=1, #save model every 5 epochs, and best model always saved in addition
#     num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
# )
##vgg11_bn
print("vgg 11 batch normalised")
model_vgg11_bn.train(
    train_df = train_df,
    validation_df = validation_df,
    save_path='../saved_models/t8c_150gc/vgg11_B32/', #trained model will be saved here
    # save_path='./model/vgg11_bn/', #trained model will be saved here
    # save_path='./binary_train/', #trained model will be saved here
    epochs=30,
    batch_size=32,
    save_interval=1, #save model every 5 epochs, and best model always saved in addition
    num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
)

## densenet 121
print("densenet 121")
model_densenet121.train(
    train_df = train_df,
    validation_df = validation_df,
    save_path='../saved_models/t8c_150gc/densenet121_B32/', #trained model will be saved here
    # save_path='./binary_train/', #trained model will be saved here
    epochs=30,
    batch_size=32,
    save_interval=1, #save model every 5 epochs, and best model always saved in addition
    num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
)
##resnet 50
print("resnet 50")
model_resnet50.train(
    train_df = train_df,
    validation_df = validation_df,
    save_path='../saved_models/t8c_150gc/resnet50_B32/', #trained model will be saved here
    # save_path='./binary_train/', #trained model will be saved here
    epochs=30,
    batch_size=32,
    save_interval=1, #save model every 5 epochs, and best model always saved in addition
    num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
)
##resnet 152
print("resnet 152")
model_resnet152.train(
    train_df = train_df,
    validation_df = validation_df,
    save_path='../saved_models/t8c_150gc/resnet152_B32/', #trained model will be saved here
    # save_path='./binary_train/', #trained model will be saved here
    epochs=30,
    batch_size=32,
    save_interval=1, #save model every 5 epochs, and best model always saved in addition
    num_workers=0 #eg 4 for 4 CPU processes, 0 for only root process
)

##MAP is mean average precision



#9. Plot loss history
    #from each epoch, to check loss is declining
    #loss should decline as model learns, but may have ups and downs
#Resnet18
# plt.scatter(model_resnet18.loss_hist.keys(), model_resnet18.loss_hist.values())
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Resnet 18 loss values')
# #model.loss_hist.keys() #gives dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# #plt.show()
# ##model_resnet18.loss_hist.values() #dict_values([0.38941222, 0.326995, 0.30710572, 0.30101994, 0.29301676, 0.28612334, 0.2761467, 0.27732074, 0.27261582, 0.2681794])
# plt.savefig("Graph_resnet18_loss_history.png")
print("resnet 18")
R18_loss_values = model_resnet18.loss_hist.values()
print(R18_loss_values)

# # #EfficientNet_b0
# # plt.scatter(model_EN_b0.loss_hist.keys(), model_EN_b0.loss_hist.values())
# # plt.xlabel('epoch')
# # plt.ylabel('loss')
# # plt.title('Efficient Net b0 loss values')
# # plt.savefig("Graph_efficientnet_b0_loss_history.png")
# print("efficient net b0")
# Eb0_loss_values = model_EN_b0.loss_hist.values()
# print(Eb0_loss_values)

# # #function for plot
# # def plot_loss(MyModel = model_resnet18, save_name = "Graph_resnet18_loss_history.png"):
# #     plt.scatter(MyModel.loss_hist.keys(), MyModel.loss_hist.values())
# #     plt.xlabel('epoch')
# #     plt.ylabel('loss')
# #     plt.title('Loss values')
# #     #plt.title('Efficient Net b0 loss values')
# #     plt.savefig(save_name)

# print("efficient net b4")
# Eb4_loss_values = model_EN_b4.loss_hist.values()
# print(Eb4_loss_values)

print("vgg 11 batch normalised")
VGG_loss_values = model_vgg11_bn.loss_hist.values()
print(VGG_loss_values)

print("densenet 121")
D121_loss_values = model_densenet121.loss_hist.values()
print(D121_loss_values)

print("resnet 50")
R50_loss_values = model_resnet50.loss_hist.values()
print(R50_loss_values)

print("resnet 152")
R152_loss_values = model_resnet152.loss_hist.values()
print(R152_loss_values)


#======================================================================================================================
#---------------------------------------------------

#10. Getting predictions on gibbon test data - for resnet18

#10a) Making functions for examining model prediction

#Making functions
    #starting with CNN model
    #getting recall, precision f1 scores

def predict_clips(MyModel):
#def predict_clips(MyModel = model_resnet18):
    #1. Making predicted values table
    #a) Getting predictions for 4 categories
    #creating dataframe
    test_predict_pd = pd.DataFrame(test_df.index, columns=["filename", "background", "gc", "mm", "sc"])
    #predicting test clips
    for i in list(range(0,test_df.shape[0])):
        predict_i = MyModel.predict([test_df.index[i]], split_files_into_clips=False)
        test_predict_pd.iloc[i, 1:5] = predict_i.values
        # if i % 100 == 0:
        #     print("i = {}".format(i))
    test_predict_pd.head()
    #                                            filename background        gc        mm        sc
    # 0  ./Audio_Label_194/label_groups/mm/2594_mm_D3.WAV  -3.242557 -7.063813 -0.501232 -0.127289
    # 1   ./Audio_Label_194/label_groups/mm/292_mm_D1.WAV   0.405559 -4.900129 -2.349729 -0.811296
    # 2  ./Audio_Label_194/label_groups/sc/7313_sc_D2.WAV  -4.060351 -1.094138 -5.030879  0.899815
    # 3  ./Audio_Label_194/label_groups/sc/6147_sc_D9.WAV   0.185952 -4.912228 -4.693875 -0.424332
    # 4  ./Audio_Label_194/label_groups/sc/4397_sc_D8.WAV  -1.228635 -7.382144 -3.378205  0.983417
    return(test_predict_pd)



def get_f1(predictions):
#def get_f1(predictions = test_predict_pd):
    #b) Adding label group column
    test_predict_label = predictions.copy()
    test_predict_label["label"] = "NaN"
    for i in list(range(0,test_predict_label.shape[0])):
        if test_df.iloc[i,0] == 1: #background
            test_predict_label.iloc[i,5] = "background" #label column
        elif test_df.iloc[i,1] == 1: #gc
            test_predict_label.iloc[i,5] = "gc"
        elif test_df.iloc[i,2] == 1: #mm
            test_predict_label.iloc[i,5] = "mm"
        elif test_df.iloc[i,3] == 1: #sc
            test_predict_label.iloc[i,5] = "sc"
    #test_predict_label.tail()

    #c) Adding predicted group column
    test_pred_4groups = test_predict_label[['background', 'gc', 'mm', 'sc']].copy()
    max_true = test_pred_4groups.eq(test_pred_4groups.max(axis=1), axis=0)
    test_predict_label["predict_group"] = max_true.dot(max_true.columns)
    test_predict_label.head()
    #                                            filename background        gc        mm        sc label predict_group
    # 0  ./Audio_Label_194/label_groups/mm/2594_mm_D3.WAV  -3.242557 -7.063813 -0.501232 -0.127289    mm            sc
    # 1   ./Audio_Label_194/label_groups/mm/292_mm_D1.WAV   0.405559 -4.900129 -2.349729 -0.811296    mm    background
    # 2  ./Audio_Label_194/label_groups/sc/7313_sc_D2.WAV  -4.060351 -1.094138 -5.030879  0.899815    sc            sc
    # 3  ./Audio_Label_194/label_groups/sc/6147_sc_D9.WAV   0.185952 -4.912228 -4.693875 -0.424332    sc    background
    # 4  ./Audio_Label_194/label_groups/sc/4397_sc_D8.WAV  -1.228635 -7.382144 -3.378205  0.983417    sc            sc

    #d) saving table as csv
    #test_predict_label.to_csv("Predictions_test_clips_2nd,_resnet18_E20_B32_random_split.csv")
    #test_predict_label.shape #(891, 7)


    #6. Examining predictions
    # #a) load prediction table - if not already loaded
    # test_predict_label = pd.read_csv("Predictions_test_clips,_resnet18_E10_B32_random_split,_from_t2b.csv")

    #b) counting labels by group
    test_label_4g_total = test_predict_label.groupby(['label'], as_index=False).size()
    print(test_label_4g_total)
    #         label  size
    # 0  background   464
    # 1          gc    22
    # 2          mm   113
    # 3          sc   292
    ## 891 test clips in total

    #c) counting predictions by group
    test_predict_4g_total = test_predict_label.groupby(['predict_group'], as_index=False).size()
    print(test_predict_4g_total)
    #   predict_group  size
    # 0    background   490
    # 1            gc     2
    # 2            mm    33
    # 3            sc   366

    #d) counting predictions and labels
    count_lab_pred = test_predict_label.groupby(['label','predict_group'], as_index=False).size()
    #print(count_lab_pred)
    #          label predict_group  size
    # 0   background    background   424
    # 1   background            gc     1
    # 2   background            mm     8
    # 3   background            sc    31
    # 4           gc    background     3
    # 5           gc            mm     2
    # 6           gc            sc    17
    # 7           mm    background    14
    # 8           mm            mm    22
    # 9           mm            sc    77
    # 10          sc    background    49
    # 11          sc            gc     1
    # 12          sc            mm     1
    # 13          sc            sc   241

    #e) calculating recall and precision
    #setting up dataframe
    count_lab_pred2 = count_lab_pred.copy()
    count_lab_pred2["lp_true_over_lab"] = "NaN" #recall
    count_lab_pred2["lp_true_over_pred"]  = "NaN" #precision
    #calculating label_prediction over total_label
        #eg for gc lab gc pred, means 25 clips from gc group were identified as gc out of 113 gc clips.
    for i in list(range(0, count_lab_pred2.shape[0])): #over 9 rows
        if count_lab_pred2.iloc[i]["label"] == count_lab_pred2.iloc[i]["predict_group"]: #for predict name being label
            ##recall - divides by label_category_total:
            label_name_i = count_lab_pred2.iloc[i]["label"]
            label_table_row_number_i = test_label_4g_total.loc[test_label_4g_total["label"] == label_name_i].index[0]
            label_count_i = test_label_4g_total.loc[test_label_4g_total["label"] == label_name_i]["size"][label_table_row_number_i]
            count_lab_pred2.at[i, "lp_true_over_lab"] = (count_lab_pred2.iloc[i]["size"]/label_count_i)*100 ##recall
            #precision - divides by predict_category_total:
            predict_name_i = count_lab_pred2.iloc[i]["predict_group"]
            predict_table_row_number_i = test_predict_4g_total.loc[test_predict_4g_total["predict_group"] == predict_name_i].index[0]
            predict_count_i = test_predict_4g_total.loc[test_predict_4g_total["predict_group"] == predict_name_i]["size"][predict_table_row_number_i]
            count_lab_pred2.at[i, "lp_true_over_pred"] = (count_lab_pred2.iloc[i]["size"]/predict_count_i)*100 ##precision
    #recall example
        #gc 25/(25+87+1) =  25/113 * 100 = 22.12% of gc labels were correctly classified
    #precision example
        #gc 25/(25+1200+114+628) = 25/1967 *100 = 1.27% of gc predictions were correct
    print(count_lab_pred2)
    #          label predict_group  size lp_true_over_lab lp_true_over_pred
    # 0   background    background   424         91.37931         86.530612
    # 1   background            gc     1              NaN               NaN
    # 2   background            mm     8              NaN               NaN
    # 3   background            sc    31              NaN               NaN
    # 4           gc    background     3              NaN               NaN
    # 5           gc            mm     2              NaN               NaN
    # 6           gc            sc    17              NaN               NaN
    # 7           mm    background    14              NaN               NaN
    # 8           mm            mm    22        19.469027         66.666667
    # 9           mm            sc    77              NaN               NaN
    # 10          sc    background    49              NaN               NaN
    # 11          sc            gc     1              NaN               NaN
    # 12          sc            mm     1              NaN               NaN
    # 13          sc            sc   241        82.534247         65.846995

    #f) summarizing recall and precision
        #recall - over labels
        #precision - over predictions
    #i. Creating table
    # import numpy as np
    # summary_perf = pd.DataFrame(index = np.arange(4), columns=["category", "recall_lab", "precision_pred"])
    summary_perf = pd.DataFrame({"category":['background','gc','mm','sc']})
    summary_perf["recall_lab"] = "NaN"
    summary_perf["precision_pred"] = "NaN"
    #      category recall_lab precision_pred
    # 0  background        NaN            NaN
    # 1          gc        NaN            NaN
    # 2          mm        NaN            NaN
    # 3          sc        NaN            NaN
    #ii. Adding recall and precision
    for i in list(range(0, count_lab_pred2.shape[0])): #over 9 rows
        if count_lab_pred2.iloc[i]["label"] == count_lab_pred2.iloc[i]["predict_group"]: #for predict name being label
            category_name_i = count_lab_pred2.iloc[i]["label"]
            ##recall - divides by label_category_total:
            recall_i = count_lab_pred2.at[i, "lp_true_over_lab"]
            ##precision - divides by predict_category_total:
            precision_i = count_lab_pred2.at[i, "lp_true_over_pred"]
            #adding to perf table
            SP_table_row_number_i = summary_perf.loc[summary_perf["category"] == category_name_i].index[0]
            summary_perf.at[SP_table_row_number_i, "recall_lab"] = recall_i
            summary_perf.at[SP_table_row_number_i, "precision_pred"] = precision_i
    #remaining category have no correct matches, therefore 0% recall and 0% precision
    for j in list(range(0, summary_perf.shape[0])): #over 4 rows
            if summary_perf.loc[j,"recall_lab"] == 'NaN':
                summary_perf.at[j, "recall_lab"] = 0
                summary_perf.at[j, "precision_pred"] = 0
    #printing table
    #print(summary_perf)
        #with rounding to 2dp
        #      category recall_lab precision_pred
        # 0  background      91.38          86.53
        # 1          gc          0              0
        # 2          mm      19.47          66.67
        # 3          sc      82.53          65.85
    #      category recall_lab precision_pred
    # 0  background   91.37931      86.530612
    # 1          gc          0              0
    # 2          mm  19.469027      66.666667
    # 3          sc  82.534247      65.846995

    #7. F1 score
        #f1 score = (2*precision*recall)/(precision + recall)
    summary_perf2 = summary_perf.copy()
    summary_perf2["f1_score"] = "Empty"
    for j in list(range(0, summary_perf2.shape[0])): #over 4 rows
        precision_i = summary_perf2.loc[j, "recall_lab"]
        recall_i = summary_perf2.loc[j, "precision_pred"]
        if (precision_i + recall_i != 0):
            f1_i = (2*precision_i*recall_i)/(precision_i + recall_i)
        else:
            f1_i = "NaN"
        summary_perf2.at[j, "f1_score"] = f1_i
    #print(summary_perf2)
    #      category recall_lab precision_pred   f1_score
    # 0  background   91.37931      86.530612  88.888889
    # 1          gc          0              0        NaN
    # 2          mm  19.469027      66.666667  30.136986
    # 3          sc  82.534247      65.846995   73.25228
    return(summary_perf2)



#b) Examining predictions, from last epoch
print("resnet-18")
R18_predict = predict_clips(MyModel = model_resnet18)
R18_predict.to_csv('../results/Predictions_t8c/Last_epoch_resnet18,_predictions_raw_t8c,_5s_loc_150gc.csv')
#R18_predict.to_csv('../results/Predictions_t8c/Predictions_raw_t8c_last_epoch,_resnet18_5s_loc_150gc.csv')
#print(R18_predict.head())
R18_f1 = get_f1(predictions = R18_predict)
print(R18_f1)
        # model_resnet18 = CNN('resnet18', classes=classes, sample_duration=5.0)
        # model_EN_b0 = CNN('efficientnet_b0', classes=classes, sample_duration=5.0)
        # model_vgg11_bn = CNN('vgg11_bn', classes=classes, sample_duration=5.0)
        # model_EN_b4 = CNN('efficientnet_b4', classes=classes, sample_duration=5.0)

# print("efficient net b0")
# EN_b0_predict = predict_clips(MyModel = model_EN_b0)
# print(EN_b0_predict.head())
# EN_b0_f1 = get_f1(predictions = EN_b0_predict)
# print(EN_b0_f1)

# print("efficient net b4")
# EN_b4_predict = predict_clips(MyModel = model_EN_b4)
# print(EN_b4_predict.head())
# EN_b4_f1 = get_f1(predictions = EN_b4_predict)
# print(EN_b4_f1)

print("vgg11 bn")
vgg11_bn_predict = predict_clips(MyModel = model_vgg11_bn)
vgg11_bn_predict.to_csv('../results/Predictions_t8c/Last_epoch_vgg11_bn,_predictions_raw_t8c,_5s_loc_150gc.csv')
#vgg11_bn_predict.to_csv('../results/Predictions_t8c/Predictions_raw_t8c_last_epoch,_vgg11bn_5s_loc_150gc.csv')
#print(vgg11_bn_predict.head())
vgg11_bn_f1 = get_f1(predictions = vgg11_bn_predict)
print(vgg11_bn_f1)

#----
print("densenet 121")
D121_predict = predict_clips(MyModel = model_densenet121)
D121_predict.to_csv('../results/Predictions_t8c/Last_epoch_densenet121,_predictions_raw_t8c,_5s_loc_150gc.csv')
D121_f1 = get_f1(predictions = D121_predict)
print(D121_f1)

print("resnet-50")
R50_predict = predict_clips(MyModel = model_resnet50)
R50_predict.to_csv('../results/Predictions_t8c/Last_epoch_resnet50,_predictions_raw_t8c,_5s_loc_150gc.csv')
#R50_predict.to_csv('../results/Predictions_t8c/Predictions_raw_t8c_last_epoch,_resnet50_5s_loc_150gc.csv')
R50_f1 = get_f1(predictions = R50_predict)
print(R50_f1)

print("resnet-152")
R152_predict = predict_clips(MyModel = model_resnet152)
R152_predict.to_csv('../results/Predictions_t8c/Last_epoch_resnet152,_predictions_raw_t8c,_5s_loc_150gc.csv')
#R152_predict.to_csv('../results/Predictions_t8c/Predictions_raw_t8c_last_epoch,_resnet152_5s_loc_150gc.csv')
R152_f1 = get_f1(predictions = R152_predict)
print(R152_f1)



#c) Examining predictions from epoch with lowest validation score

#loading these models
model_resnet18_LVS = load_model('../saved_models/t8c_150gc/resnet18_B32/best.model')
model_vgg11_LVS = load_model('../saved_models/t8c_150gc/vgg11_B32/best.mode')
model_densenet121_LVS = load_model('../saved_models/t8c_150gc/densenet121_B32/best.model')
model_resnet50_LVS = load_model('../saved_models/t8c_150gc/resnet50_B32/best.model')
model_resnet152_LVS = load_model('../saved_models/t8c_150gc/resnet152_B32/best.model')
##model_resnet18_LVS = load_model('6b_pytorch_150gc/R18_150gc_model/best.model')


print("\n", "Lowest_validation_epoch")

print("resnet-18")
R18_predict = predict_clips(MyModel = model_resnet18)
R18_predict.to_csv('../results/Predictions_t8c/Lowest_validation_epoch_resnet18,_predictions_raw_t8c,_5s_loc_150gc.csv')
R18_f1 = get_f1(predictions = R18_predict)
print(R18_f1)

print("vgg11 bn")
vgg11_bn_predict = predict_clips(MyModel = model_vgg11_bn)
vgg11_bn_predict.to_csv('../results/Predictions_t8c/Lowest_validation_epoch_vgg11_bn,_predictions_raw_t8c,_5s_loc_150gc.csv')
vgg11_bn_f1 = get_f1(predictions = vgg11_bn_predict)
print(vgg11_bn_f1)

print("densenet 121")
D121_predict = predict_clips(MyModel = model_densenet121)
D121_predict.to_csv('../results/Predictions_t8c/Lowest_validation_epoch_epoch_densenet121,_predictions_raw_t8c,_5s_loc_150gc.csv')
D121_f1 = get_f1(predictions = D121_predict)
print(D121_f1)

print("resnet-50")
R50_predict = predict_clips(MyModel = model_resnet50)
R50_predict.to_csv('../results/Predictions_t8c/Lowest_validation_epoch_epoch_resnet50,_predictions_raw_t8c,_5s_loc_150gc.csv')
R50_f1 = get_f1(predictions = R50_predict)
print(R50_f1)

print("resnet-152")
R152_predict = predict_clips(MyModel = model_resnet152)
R152_predict.to_csv('../results/Predictions_t8c/Lowest_validation_epoch_epoch_resnet152,_predictions_raw_t8c,_5s_loc_150gc.csv')
R152_f1 = get_f1(predictions = R152_predict)
print(R152_f1)



