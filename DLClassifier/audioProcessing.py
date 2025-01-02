import matlab.engine
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from timeit import default_timer as timer
from datetime import datetime 

from obspy import read
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
import scipy.signal as sps



def resize(m, x, y):
    m = np.asarray(abs(m))
    x = 64
    y=64

    m_mp = np.zeros([y, x])
    ht = m.shape[0]
    l = m.shape[1]

    ly = np.int32(ht/y)
    lx = np.int32(l/x)

    remainder_x = int((l/x-lx)*x); #unused columns in original image if each column of the new image has only been sampled from lx columns 
    remainder_y = int((ht/y-ly)*y);#unused rows


    inds_x_extra_sample = np.round(np.linspace(0,x-1, remainder_x)); #positions where the extra columns will be redistributed
    inds_y_extra_sample = np.round(np.linspace(0, y-1, remainder_y)); #positions where the extra rows will be redistributed


    inds_x = np.zeros([x, 1])+lx; #starting indices if all bins were of length lx
    if len(inds_x_extra_sample)>0:
        inds_x[list(inds_x_extra_sample.astype(int))] =lx+1;#add the extra columns

    fins_x = np.cumsum(inds_x).astype(int); #the cummulative sum of lengths marks the end of each horizontal bin
    inis_x = [0] + list(fins_x[:x-1]) #start indices of horizontal bins

    #repeat for rows:
    inds_y = np.zeros([y, 1])+ly;
    if len(inds_y_extra_sample)>0:
        inds_y[list(inds_y_extra_sample.astype(int))] =ly+1; #add the extra columns
    fins_y = np.cumsum(inds_y).astype(int); #the cummulative sum of lengths marks the end of each horizontal bin
    inis_y = [0] + list(fins_y[:y-1]) #start indices of horizontal bins



    for i in range(x):
        for j in range(y):
            ini_x = inis_x[i]; #find the starting horizontal index of bin 
            fin_x = fins_x[i]+1; #end horizontal index 
            ini_y = inis_y[j]; #starting vertical index of bin
            fin_y = fins_y[j]+1; #end vertical index of bin
            clip = m[ini_y:fin_y, ini_x:fin_x]; #crop bin that will be downsampled to one pixel

            m_mp[y-j-1, i] = np.max(clip); #Find max value of the bin and store it in new matrix
    return m_mp




def prepare_images (eng, wav_path):
    '''
    This function calls the MatLab function getSpectrogram to obtain all the spectrograms for non-overlapping 5s segments from the wav_file which path is given as a parameter
    '''

 #   print(wav_path) #Print path (for debugging purposes)

    imgs=[] #Initialize list which will contain spectrograms ready to feed NN


    spec=eng.getSpectrogram(wav_path, 5, 48, 64, 64)#get spectrograms (calling MATLAB function)

    if spec != 0: #Check that the function didn't return error

        spec_array = np.array(spec._data).reshape(spec.size[::-1]).T #convert spectrogram to numpy array
  #      print(spec_array.shape)
        for i in range(48): # Iterate over 5s segments, and append them to imgs list
            clip = spec_array[:, :, i] 
            imgs.append(clip)
       
        return (imgs)
    return (0) #If MatLab function returns 0 (error code), pass it to the next function



def process_file_spec(model, eng, wav_path):
    '''
    This function obtains the model predictions for all non-overlapping 5s segments of a wav file (path given as an argument) containing sperm whale clicks
    '''

    imgs=prepare_images(eng, wav_path)   #obtain list of all the 5s spectrograms 
    
    if imgs != 0: # Check that there wasn't an error code
        predictions=[] # Initialize list of predictions
        for img in imgs: 
            #Prepare image for the CNN
            img=torch.tensor(img)
            img=torch.reshape(img, [1, 1, 64, 64])  

            #run detector on image
            with torch.no_grad():
                _,pred_y = model(img.float())
                a=round(float(pred_y[0][1]),4) # Select class probability of the 5s segment containing spermwhale clicks, and append it to predictions vector
                predictions.append(a)
        return(predictions) #Return predictions vector
    return(0) #If there was an error with the matlab function, pass it onwards


def prepare_numpies(wavpath):

    st = read(wavpath, format="wav")
    fs = int(st[0].stats.sampling_rate)
    w = st[0].data
    l = 5*fs
    
    new_fs = 48000
    number_of_samples = round(l * float(new_fs) / fs)
    dt = 1/new_fs
    f_min = 1000
    f_max = 20000
    scarle = 128
    cycles=8

    numpies = []

    for segment in range(47):
        ini = (segment)*l #start sample
        fin = ini+l #end sample
        chunk = w[ini:fin]
        chunk = sps.resample(chunk, number_of_samples)
        scalogram = cwt(chunk, dt, cycles, f_min, f_max, nf=scarle)
        resized_scalo = resize(scalogram[:, 400:-400], 64, 64)
        numpies.append(resized_scalo)

    #Do last segment
    fin = len(w)
    ini = fin-l
    chunk = w[ini:fin]; 
    chunk = sps.resample(chunk, number_of_samples)
    scalogram = cwt(chunk, dt, cycles, f_min, f_max, nf=scarle)
    resized_scalo = resize(scalogram[:, 400:-400], 64, 64)
    numpies.append(resized_scalo)
    print ('Length numpies', len(numpies))
    return(numpies)


def process_file_scalo(model, wavpath):

    numpies = prepare_numpies(wavpath)
    predictions = [] # initialize list of predictions
    for scalo in numpies:
        scalo = torch.from_numpy(scalo)
        scalo=torch.reshape(scalo, [1, 1, 64, 64])

        #run detector on image
        with torch.no_grad():
            _,pred_y = model(scalo.float())
            a=round(float(pred_y[0][1]),4) # Select class probability of the 5s segment containing spermwhale clicks, and append it to predictions vector
            predictions.append(a)
    print('Length prediticions', len(predictions)) 
    return(predictions) #Return predictions vector


if __name__=="__main__":

    pass
    #run_model(input_path, output_path)


