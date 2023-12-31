from __future__ import division

import glob

# import scipy.misc as sc
import cv2
import numpy as np
import scipy.io as sio



# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = 'dataset_isic18/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 129 training samples
Data_train_2018    = np.zeros([129, height, width, channels])
Label_train_2018   = np.zeros([129, height, width])

print('Reading ISIC 2018')
for idx in range(len(Tr_list)):
    print(idx+1)

    if(idx + 1> 129):
        break
    img = cv2.imread(Tr_list[idx])
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.double)

    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC2018_Task1_Training_GroundTruth/' + b +'_segmentation.png')    
    img2 = cv2.imread(add)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = img2.astype(np.double)
    Label_train_2018[idx, :,:] = img2
         
print('Reading ISIC 2018 finished')

################################################################ Make the train and test sets ########################################    
# We consider 113 samples for training, 16 samples for validation and 520 samples for testing

Train_img      = Data_train_2018[0:113,:,:,:]
Validation_img = Data_train_2018[113:113+16,:,:,:]
Test_img       = Data_train_2018[113+16:129,:,:,:]

Train_mask      = Label_train_2018[0:113,:,:]
Validation_mask = Label_train_2018[113:113+16,:,:]
Test_mask       = Label_train_2018[113+16:129,:,:]

np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)


