#Predict whole folder at a particular location

#1. Libraries
from opensoundscape import load_model #saved vgg11
import opensoundscape
from glob import glob
import pandas as pd


#making function
def make_predict_table(input_sound, csv_path):

    #2. Loading saved CNN
    #a) loading vgg11 B32
    model_vgg11 = load_model('vgg11_B32_choosen.model')
    #b). Enabling 16000Hz sample rate
        #doing bypass so runs on 16000Hz sample rate audio
    model_vgg11.preprocessor.pipeline.bandpass.bypass=True

    #3. Getting audio file paths
    audio_files = glob(input_sound)
    #audio_files = glob('China_side/02/*.WAV')

    #4. Generating predictions
        #non-overlapping
    scores = model_vgg11.predict(audio_files)
    ###scores.to_csv(csv_path, index=False)
#    scores.to_csv(csv_path) #adds file name, start_time, end_time
    ###scores.to_csv('../results/Location_predictions_2021_China_side_A2.csv')

    #5. Adding column - category with highest prediction
    #a) creating function
    def PG_row_max(test_predict_label):
        """Adding predicted group column, 
        based on which of four category has largest predicted value"""
        test_pred_4groups = test_predict_label[['background', 'gc', 'mm', 'sc']].copy()
        max_true = test_pred_4groups.eq(test_pred_4groups.max(axis=1), axis=0)
        test_predict_label["predict_group"] = max_true.dot(max_true.columns)
        #test_predict_label.head()
        return(test_predict_label)
    #b) running function
    predict_table = PG_row_max(scores)
    # #c) saving csv
    # predict_table.to_csv(csv_path, index=False)
    #predict_table.to_csv('../results/Category_predict_table,_China_side_02.csv', index=False)
#    print(predict_table.head())

    #6. Adding time in minutes columns
#    predict_table['start_mins'] = predict_table['start_time']/60
#    predict_table['end_mins'] = predict_table['end_time']/60

    #7. Adding location and file name columns
        #For splitting eg China_side/02/20211221_213000.WAV 
#    predict_table['location'] = predict_table['file'].str.split('/', expand =True)[1]
#    predict_table['file_time'] = predict_table['file'].str.split('/', expand =True)[2]

    #8. saving csv
    predict_table.to_csv(csv_path)

    #9. examining results
    #a) counting number of predictions in each category
    category_count = predict_table.groupby(['predict_group'], as_index=False).size()
    print(category_count)
    #b) counting number at each audio file
#    category_by_file_count = predict_table.groupby(['predict_group','audio_file_name'], as_index=False).size()
#    print(category_by_file_count)
