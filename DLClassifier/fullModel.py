import matlab.engine
import numpy as np
import pandas as pd

from clickDetector import LeNet5
from fileClassifier import LinearNet
from audioProcessing import process_file

import torch
import torchvision.transforms as transforms
from timeit import default_timer as timer
from datetime import datetime 
import sys
import os.path

#Config Parameters:
import config as config

ROOT_DIR = config.ROOT_DIR
HARD_DRIVE_PATH = config.HARD_DRIVE_PATH


loc = sys.argv[1]






def run(site):
   
    eng = matlab.engine.start_matlab() #Initialize MatLab engine
    eng.addpath(config.MATLAB_PATH,nargout=0)

    #Load trained CNN model
    CNN_model = LeNet5(config.CLASSIFIER_IN)
    CNN_model.load_state_dict(torch.load(ROOT_DIR+"clickDetectorFiles/clickDetector_model_weights.pth"))
    CNN_model.eval()

    #Load trained LinearNN model
    LinearNN_model = LinearNet()
    LinearNN_model.load_state_dict(torch.load(ROOT_DIR + "fileClassifierFiles/LinearNet_model_weights.pth"))
    LinearNN_model.eval()

    #Output table path
    output_table=ROOT_DIR+"outputs/SW20_probabilities_"+site+".xlsx"
    #If an output table already exists, it means that the model has stopped miway for some reason. Find first empty index and go from there
    if os.path.isfile(output_table):
        df =pd.read_excel(output_table)
        x = list(df.DetectorOutput)
        nan_inds = np.argwhere(np.isnan(x))
        ini=nan_inds[0, 0]-1
    else: #If it doesn't, load empty table with all training files
        table=ROOT_DIR+"inputs/"+loc+".xlsx"
        df = pd.read_excel(table)
        colNames = ["UTC", "File", "Path", "Location"]
        df = df.loc[:, colNames]
        df["DetectorOutput"]=["" for i in range(len(df))] #Initialize DetectorOutput column
        ini=0

    names = ["ClassSegment"+str(i+1) for i in range(48)] # Column names where CNN output is stored


    for i, row in df.iloc[ini:len(df)].iterrows(): #Iterate over empty rows of table
        path=row['Path']
        path = path.replace('/Volumes/Balearic_SW/', HARD_DRIVE_PATH)
        filepath=path + row['File']

        #print(filepath)
        predictions = process_file(CNN_model, eng, filepath) #CNN predictions
        if predictions==0 or predictions[0]==-1 or len(predictions)<48: # Check that there wasn't an error with the CNN
            final_output = -1 
        else: 
            #If it all went well, convert output to torch tensor and send it to the sequential NN 
            seq=torch.tensor(np.array(predictions))
            seq = seq.float()
            _,s = LinearNN_model(seq)
            final_output = round(float(s[1]),3) # Final detector output

        df.DetectorOutput[i] =final_output
        #print(len(predictions))
           # for j in range(len(names)):
                #print(i, j)
            #    df_out.loc[i, names[j]]=predictions[j]

        # Every 100 rows, save dataframe to excel file
        if i%100==0:
            print(f'{datetime.now().time().replace(microsecond=0)} --- ')
            print("Saving up to row "+str(i))
            df.to_excel(output_table)
    df.to_excel(output_table)




if __name__=="__main__":

    run(loc) 
    #run_on_AM()
