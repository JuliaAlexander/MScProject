#Example of applying CNN classifier on raw audio recordings
	#In this example, the classifier generates predictions from some Vietnam Phase 1-2 main survey folders

#Importing function
from Location_predictions_function import *

folders_to_predict = ["AU02","AU03","AU04","AU05"]

for name in folders_to_predict:
    input = str("U_Vietnam_Phase_1-2/")+name+str("/*.WAV")
    output = str("../results/U_Vietnam_Phase_1-2/")+name+str("_Vietnam_P1-2_predictions.csv")
    #make_predict_table(input_sound='China_side/02/*.WAV', csv_path='../results/A2_China_predictions.csv')
    make_predict_table(input_sound=input, csv_path=output)
