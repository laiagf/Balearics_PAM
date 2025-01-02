import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io



import config


### TRAIN ###
def train(dataloader, model, criterion, optimizer, device, training=True):
    '''
    This function runs a train (default) or validation (when training=False) step of the training loop
    '''
    
    #Set model in either training or evaluation mode
    if training:
        model.train()
    else:
        model.eval()
    
    #Set running loss to 0
    running_loss = 0
    
    for X, y_true in dataloader: #iterate over batches of dataloader
        if training:
            optimizer.zero_grad() #initialize gradients of optimizer
        
        #Send object and correct label to device
        #X = X.float() #lgar change 0712
        X = X.to(device) 
        y_true = y_true.to(device) 
    
        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
        
        if training: #Run a backward pass if we're in training mode
            # Backward pass
            loss.backward()
            optimizer.step() 
        
    epoch_loss = running_loss / len(dataloader.dataset) #Compute total loss for epoch
    
    #Return model, optimizer(if training) and epoch_loss
    if training:
        return model, optimizer, epoch_loss
    else:
        return model, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    This function calls the train function on validation model
    ''' 
        
    return train(valid_loader, model, criterion, False, device, False)



def get_accuracy(model, data_loader, device):
    '''
    This function computes the accuracy of the model predictions over a data_loader, as well as false positive and false negative rates
    '''
    
    #Initialize counts to 0
    tp = 0 #cummulative number of true positive classifications
    tn = 0 #cummulative number of true negative classifications
    fp = 0 #cummulative number of false positive classifications
    fn = 0 #cummulative number of false negative classifications
    n = 0 #cummulative number of objects checked
    
    
    with torch.no_grad(): #Set model in training mode
        model.eval()
        for X, y_true in data_loader: #Iterate over data_loader

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X) #Run model
            _, predicted_labels = torch.max(y_prob, 1) #Get predicted labels
            
            n += y_true.size(0) #add number of elements in dataloader batch to n
            #add numbers of correct predictions, false positive and false negatives
            tp += ((predicted_labels ==y_true)*(predicted_labels==1)).sum() 
            tn += ((predicted_labels ==y_true)*(predicted_labels==0)).sum()
            fp += ((predicted_labels !=y_true)*(predicted_labels==1)).sum() 
            fn += ((predicted_labels !=y_true)*(predicted_labels==0)).sum()
   
    acc = 100*(tp.float()+tn.float())/n #accuracy=(tp+tn)*100/(tp+tn+fp+fn)   
    FDR = fp.float()/(fp+tp)
    Recall = tp.float()/(tp+fn)
    Precision = tp.float()/(tp+fp)
    #Return accuracy, FDR, recall and precision 
    return (acc, FDR, Recall, Precision)



def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, output_dir, print_every=1):
    '''
    This function runs the entire training loop for the model
    '''
    
    # Prepare vectors for storing metrics
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    train_fpositives = []
    train_fnegatives=[]
    valid_fpositives=[]
    valid_fnegatives=[]  

 
    #Run a first validation pass before starting to train the model, and store and print results
    with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            model, train_loss = validate(train_loader, model, criterion, device)
    valid_acc, valid_fdr, valid_rec, valid_pres = get_accuracy(model, valid_loader, device=device)
    train_acc, train_fdr, train_rec, train_pres = get_accuracy(model, train_loader, device=device)
    train_losses.append(train_loss)
    #train_accuracies.append(train_acc*100)
    valid_losses.append(valid_loss)
    #valid_accuracies.append(valid_acc*100)    
    #train_fpositives.append(train_fpos*100) 
    #train_fnegatives.append(train_fneg*100) 
    #valid_fpositives.append(valid_fpos*100) 
    #valid_fnegatives.append(valid_fneg*100) 
    print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  'Epoch: -1\n'
                  f'Traning set:\t'
                  f'Loss: {train_loss:.4f}\t'
                  f'FDR: {train_fdr:.4f}\t'
                  f'Recall: {train_rec:.4f}\t'
                  f'Precision: {train_pres:.4f}\n'
                  f'Validation set:\t'
                  f'Loss: {valid_loss:.4f}\t'
                  f'FDR: {valid_fdr:.4f}\t'
                  f'Recall: {valid_rec:.4f}\t'
                  f'Precision: {valid_pres:.4f}\n')
                
    
   #               f'Valid loss: {valid_loss:.4f}\t'
    #              f'Train accuracy: {100 * train_acc:.2f}\t'
     #             f'Valid accuracy: {100 * valid_acc:.2f}\t'
     #             f'Train false positives: {100 * train_fpos:.2f}\t'
      #            f'Valid false positives: {100 * valid_fpos:.2f}\t'
       #           f'Train false negatives: {100 * train_fneg:.2f}\t'
        #          f'Valid false negatives: {100 * valid_fneg:.2f}')

    # Train model
    for epoch in range(0, epochs):

        # training step
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation step
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
        
        # Compute accuracy on training and validation set. Store and print
        
        valid_acc, valid_fdr, valid_rec, valid_pres = get_accuracy(model, valid_loader, device=device)
        train_acc, train_fdr, train_rec, train_pres = get_accuracy(model, train_loader, device=device)
        if epoch % print_every == (print_every - 1):
            

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\n'
                  f'Traning set:\t'
                  f'Loss: {train_loss:.4f}\t'
                  f'FDR: {train_fdr:.4f}\t'
                  f'Recall: {train_rec:.4f}\t'
                  f'Precision: {train_pres:.4f}\n'
                  f'Validation set:\t'
                  f'Loss: {valid_loss:.4f}\t'
                  f'FDR: {valid_fdr:.4f}\t'
                  f'Recall: {valid_rec:.4f}\t'
                  f'Precision: {valid_pres:.4f}\n')
    #Plot the losses, accuracy and rates of false positives and negatives on training and validation set over epochs 
    #plot_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies, train_fpositives, train_fnegatives, valid_fpositives, valid_fnegatives, output_dir)
    
    return model, optimizer, (train_losses, valid_losses)


def plot_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies, train_fpos, train_fneg, valid_fpos, valid_fneg, output_dir):
    '''
    This function plots accuracies, losses and false positive and negative rates for both the training and validation set over epochs
    '''
    

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)
    train_acc = np.array(train_accuracies)
    valid_acc = np.array(valid_accuracies)
    
    epochs = np.arange(-1, len(train_acc)-1)
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 18))
    
    axes[0].plot(epochs, train_losses, color='blue', label='Training loss') 
    axes[0].plot(epochs, valid_losses, color='red', label='Validation loss')
    axes[0].set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    axes[0].legend()
    
    axes[1].plot(epochs, train_acc, color='blue', label='Training accuracy') 
    axes[1].plot(epochs, valid_acc, color='red', label='Validation accuracy')
    axes[1].set(title="Accuracy over epochs", 
            xlabel='Epoch',
            ylabel='Accuracy %') 
    axes[1].legend()
    
    axes[2].plot(epochs, train_fpos, color='blue', label='Train false positives')   
    axes[2].plot(epochs, train_fneg, color='grey', label='Train false negatives')   
    axes[2].plot(epochs, valid_fpos, color='red', label='Valid false positives')   
    axes[2].plot(epochs, valid_fneg, color='yellow', label='Valid false negatives')
    axes[2].set(title="False detections over epochs",
            xlabel='Epoch',
            ylabel="False detections \%")
    axes[2].legend()   
    fig.savefig(output_dir+'accuracy.png')



