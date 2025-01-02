import numpy as np
from datetime import datetime

import matlab.engine

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io

from training_loop import training_loop

from audioProcessing import process_file_spec, process_file_scalo

DEVICE ='cpu'


#Config parameters
import config as config 

FILES_DIR = config.ROOT_DIR + "clickDetectorFiles/" 

IMG_DIR = config.ROOT_DIR + "ImageTrainingSet/"
NUMPY_DIR = 'C:/Users/lgf3/Projects/scalograms/LeNet_experiment/numpies/'

CLASSIFIER_IN = config.CLASSIFIER_IN

HARD_DRIVE_PATH = config.HARD_DRIVE_PATH



# NN training parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.0005 #0.001
BATCH_SIZE = 32
#N_EPOCHS = 22
N_EPOCHS=2



class ClickDataset(Dataset):
    '''
    Custom Dataset of downsampled spectrogram images labelled as containing or not containing sperm whale clicks
    '''
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform= transform

        
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = io.imread(img_path)
        if (len(image.shape)>2):
            image=image[:, :, 0]
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        
        if self.transform:
            image= self.transform(image)

        return (image, y_label)

class scalogramDataset(Dataset):
    '''
    Custom dataset of downsaple scalogram numpies labelled as containing or not containing sperm whale clicks
    '''
    def __init__(self, csv_file, numpy_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.numpy_dir = numpy_dir
        self.transform= transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index,0]
        npy_name = img_name[:-3] + 'npy'
        scalogram = np.load(os.path.join(self.numpy_dir, npy_name))
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
            
        if self.transform:
            scalogram= self.transform(scalogram)

        scalogram = scalogram.float()

        return (scalogram, y_label)

class LeNet5(nn.Module):
    '''
    CNN based on the LeNet5 architecture adapted to fit input images of different sizes (chosen by parameter classifier_in) and only 2 output classes (presence/absence of sperm whale clicks) 
    Activation function changed to ReLU.
    Possible input sizes:
        - 64x64 pixels: classifier_in = 9720
        - 128x32 pixels: classifier_in = 3000
    '''
    def __init__(self, CLASSIFIER_IN):
        super(LeNet5, self).__init__()
        #Convolutional part of the NN
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )
        #Linear part of the NN
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=CLASSIFIER_IN, out_features=84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=84, out_features=2),

        )
    #Forward pass function. Returns losses and class probabilities.
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs





def train():
    '''
    This function loads the data, trains the model from scratch, and stores it.
    '''
    #Instantiate the model
    torch.manual_seed(RANDOM_SEED)
    model = LeNet5(CLASSIFIER_IN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    
    #Load Data
    csv_file=FILES_DIR+"train_labelsI.csv" #cvs of img names and manual labels
    if config.FEATURE=='spec':
        dataset = ClickDataset(csv_file=csv_file, img_dir=IMG_DIR, transform=transforms.ToTensor()) #Create dataset of spectrograms and labels
    else:
        dataset = scalogramDataset(csv_file=csv_file, numpy_dir=NUMPY_DIR, transform=transforms.ToTensor()) #Create dataset of scalograms and labels
    split85=int(len(dataset)*0.85) 
    train_set, val_set = torch.utils.data.random_split(dataset,  [split85, len(dataset)-split85]) #Randomly split data so that 85% of it will be used for training and 15% for testing
    
    #Create dataloaders for sending data to the CNN   
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=True)


    #Train the model
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader,N_EPOCHS, DEVICE, FILES_DIR)
    
    #Save model weights
    if config.FEATURE=='spec':
        torch.save(model.state_dict(),FILES_DIR+"clickDetector_model_weights.pth" )
    else:
        torch.save(model.state_dict(),FILES_DIR+"clickDetector_model_weights_scalo.pth" )
    
    return model #lgar 712

def run(model):
    '''
    This function runs our trained modified LeNet model on the files listed in an input_table, and stores the CNN outputs in a table stored at output_path
    '''
    eng = matlab.engine.start_matlab() #Initialize MatLab engine
    eng.addpath(config.MATLAB_PATH, nargout=0)


    #Load model
    '''
    #lgar 712, problem loading torch so well do that
    model=LeNet5(CLASSIFIER_IN)
    
    if config.FEATURE=='spec':
        model.load_state_dict(torch.load(FILES_DIR + "clickDetector_model_weights.pth"))
    else:
        print(FILES_DIR+"clickDetector_model_weights_scalo.pth")
        checkpoint = torch.load(FILES_DIR + "clickDetector_model_weights_scalo.pth")
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.float()
    '''

    #file tables
    input_path = FILES_DIR+"testingFiles.xlsx"
    if config.FEATURE=='spec':
        output_path = FILES_DIR+"outputClickDetectorTestingFiles.xlsx"
    else:
        output_path = FILES_DIR+"outputClickDetectorTestingFilesScalo.xlsx"

    df = pd.read_excel(input_path) # read input table

    names = ["ClassSegment"+str(i+1) for i in range(48)]
    print(len(names))
    for i, row in df.iloc[1601:len(df)].iterrows(): #Iterate over wav files listed in input_table
        path=row['Path']
        path = path.replace('/Volumes/Balearic_SW/', HARD_DRIVE_PATH)
        filepath=path + row['File']

        if config.FEATURE =='spec':
            predictions = process_file_spec(model, eng, filepath) # Obtain CNN predictions for wav files
        else:
            predictions = process_file_scalo(model, filepath)

        if predictions != 0: # If no error was raised, store predictions in output table
            for j in range(len(names)):
                print(i, j)
                df.loc[i, names[j]]=predictions[j]
        if i%100==0: # Save output values every 100 iterations, and print a notification
            print(f'{datetime.now().time().replace(microsecond=0)} --- ')
            print("Saving up to row "+str(i))
            df.to_excel(output_path)
    df.to_excel(output_path) # Save output table again at the end of the loop


if __name__=="__main__":

    model = train()
    run(model)
