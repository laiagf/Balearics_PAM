import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from training_loop import training_loop

import config as config #configuration parameters

#Training Parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 8 
N_EPOCHS = 20 

DEVICE = config.DEVICE

#Neural Net Parameters
H=10 #number of neurons in hidden layer 



FILES_DIR = config.ROOT_DIR+"fileClassifierFiles/"



def prepareTrainingData():
    if config.CLASSIFIER_IN==3000:
        names = ["ClassSegment"+str(i+1) for i in range(24)]
    else:
        names = ["ClassSegment"+str(i+1) for i in range(48)]
    colNames = ["File", "SW20", "Location"]+names

    df = pd.read_excel(config.ROOT_DIR+"clickDetectorFiles/outputClickDetectorTestingFiles.xlsx")
    df = df[colNames]

    df_used = pd.read_csv(config.ROOT_DIR+"clickDetectorFiles/train_labels.csv")
    files_used = []

    for f in list(df_used["File"]):
        frev = f[::-1]
        ind=frev.find("_")
        f=frev[len(frev):ind:-1]+".wav"
        files_used.append(f)

    used = []

    for i in range(len(df)):
        filename = df.loc[i]["File"]
        if filename in files_used:
            used.append(1)
        else:
            used.append(0)

    #df["Used"] = used
    df.insert(3, "Used4Training", used)
    df.to_csv(FILES_DIR+"/trainingData.csv", index=False)

    

class SequenceDataset(Dataset):
    ''' 
    Custom Dataset class of vectors of the CNN outputs of a file manually labelled following the SW20 criteria
    '''
    def __init__(self, df, colNames, transform=None):
        self.annotations = df
        self.colNames = colNames
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        seq = torch.tensor(self.annotations[self.colNames].iloc[index])
        y_label = torch.tensor(self.annotations["SW20"].iloc[index])
        return (seq.float(), y_label)




class LinearNet(nn.Module):
    '''
    Linear NN with an input size of 48 (length of the ClickDetector outputs for a 40' file) and one hidden layer 
    Developed to classify full files from the sequences of CNN outputs according to the SW20 criteria
    '''

    def __init__(self):
        super(LinearNet, self).__init__()
        
        self.seq = nn.Sequential(
            torch.nn.Linear(in_features=48, out_features=H),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=H, out_features=2),
       )

    def forward(self, s):
        s = self.seq(s)
        probs = F.softmax(s, dim=-1)
        return (s, probs)





def train():

    '''
    This function loads the training and testing data, trains the model from scratch, and stores it.
    '''

    csv_trainingData=FILES_DIR+"trainingData.csv"


    df = pd.read_csv(csv_trainingData)
    train_df = df[df.Used4Training==1]
    val_df = df[df.Used4Training==0]

    names = ["ClassSegment"+str(i+1) for i in range(48)] #Names of columns where CNN outputs are stored


 
    train_set = SequenceDataset(df=train_df, colNames=names)
    val_set = SequenceDataset(df=val_df, colNames=names)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)  

    # Initialize model
    torch.manual_seed(RANDOM_SEED)
    model = LinearNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() 

    # Train model   
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader,N_EPOCHS, DEVICE, FILES_DIR)
    
    # Store model weights
    torch.save(model.state_dict(),FILES_DIR+"LinearNet_model_weights.pth" )



def run():
    '''
    This function runs the trained LinearNet model on the set of manually labelled files, and stores its output in the same table
    '''

    #Load linear model
    seq_model = LinearNet()
    #seq_model.load_state_dict(torch.load(experiment_dir + "seq_model_weights"+loc+".pth"))
    seq_model.load_state_dict(torch.load(FILES_DIR + "LinearNet_model_weights.pth"))
    seq_model.eval()

    #Load table of CNN outputs
    csv_file = FILES_DIR+"trainingData.csv"
    df = pd.read_csv(csv_file)
    df["DetectorOutput"] = [-1 for i in range(len(df))] #Initialize DetectorOutput column

    colNames =  ["ClassSegment"+str(i+1) for i in range(48)] #Names of columns where CNN outputs are stored

    for i in range(len(df)):
        # Find CNN predictions for file i, convert them to tensor and find the LinearNet classification
        predictions = list(df.loc[i, colNames]) 
        seq=torch.tensor(np.array(predictions))
        seq = seq.float()
        _,s = seq_model(seq)
        final_output = round(float(s[1]),3) # Final detector output
        #print(final_output)
        df.loc[i, "DetectorOutput"] = final_output
    df.to_excel(FILES_DIR+"fileClassifierOutput.xlsx", index=False)


if __name__=='__main__':
    prepareTrainingData()
    train()
    run()
