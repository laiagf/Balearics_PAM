
ROOT_DIR = "C:/Users/lgf3/Projects/Balearics_PAM/"

HARD_DRIVE_PATH = "/media/laiagarrobe/Balearic_SW/"

CLASSIFIER_IN = 9720 #9720 for 64x64 pixels input images, 3000 for 128x32 pixels

MATLAB_PATH = ROOT_DIR+"DLClassifier/MATLAB/"

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
 
FEATURE = 'scalo'

