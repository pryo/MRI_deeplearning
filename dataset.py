import os
import random

from PIL import Image
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader
#patch xmax = 149
#patch ymax =168
class DualChannelAPTDataset(Dataset):
    """load the apt.img only"""
    def __init__(self,root_dir,gz_list,kki_list,ROILogPath,
                 truth_path='idh.xlsx',transform=None, switch = 0,
                 ppms=['aptw','2ppm','mtr']):
        # switch =0 for all
        #1 for gz
        # 2 for kki
        self.ppms = ppms
        self.switch = switch
        self.root_dir = root_dir
        self.transform = transform
        self.gz_list = gz_list
        self.kki_list = kki_list
        self.ROIdf  = pd.read_csv(ROILogPath,header = None,index_col=0)
        self.dataframe = pd.read_excel(truth_path)
        #self.odd_list = [10,49,73,74,75,76,8,97,100]
        self.odd_list = [74, 76, 100]
        #todo load truth file

    def readCoor(self,ID):
        return eval(str(self.ROIdf.loc[int(ID)][1]))

    def getPatch(self,image, coor):
        # coor 0:x1 1:y1 2:x2 3:y2
        coor = [round(x) for x in coor]

        patch = image[coor[1]:coor[3], coor[0]:coor[2]]
        return patch

    def pad(self,array, xMaxDiff=150, yMaxDiff=170):
        # x 150
        # y 170
        yPadTotal = int(yMaxDiff - array.shape[0])
        if yPadTotal % 2 == 0:
            yPadSingleUp = yPadTotal / 2
            yPadSingleDown = yPadTotal / 2
        else:
            yPadSingleUp = (yPadTotal + 1) / 2 - 1
            yPadSingleDown = (yPadTotal + 1) / 2

        xPadTotal = int(xMaxDiff - array.shape[1])
        if xPadTotal % 2 == 0:
            xPadSingleUp = xPadTotal / 2
            xPadSingleDown = xPadTotal / 2
        else:
            xPadSingleUp = (xPadTotal + 1) / 2 - 1
            xPadSingleDown = (xPadTotal + 1) / 2
            # print(((yPadSingleUp,yPadSingleDown),(xPadSingleUp,xPadSingleDown)))
        paddedArray = np.pad(array,
                             ((int(yPadSingleUp), int(yPadSingleDown)),
                              (int(xPadSingleUp), int(xPadSingleDown))),
                             mode='constant')
        return paddedArray
    def __len__(self):
        if self.switch ==0:

            return len(self.kki_list+self.gz_list)
        elif self.switch==1:
            return len(self.gz_list)
        elif self.switch==2:
            return len(self.kki_list)

    def __getitem__(self,idx):
        # three channel duplicate
        id = None
        if self.switch ==0:
            id = str(idx + 2)

            while int(id) in self.odd_list:
                id = random.randint(0, len(self.kki_list+self.gz_list))+2
                # deflex the sample from odd list

        elif self.switch ==1:
            id = self.gz_list[idx]
            while int(id) in self.odd_list:
                rand_idx = random.randint(0, len(self.gz_list)-1)
                id = self.gz_list[rand_idx]
        elif self.switch ==2:
            id = self.kki_list[idx]
            while int(id) in self.odd_list:
                id = self.kki_list[random.randint(0, len(self.kki_list)-1)]
        apt_img_path = ''
        #id = str(idx+2)
        dir = os.path.join(self.root_dir,
                           id)
        apt_img = None
        id = str(id)
        ch1 = None
        ch2 = None
        ch3 = None
        patch1 = None
        patch2 = None
        patch3 = None
        if id in self.gz_list:
            patchCoor = self.readCoor(id)

            for name in os.listdir(dir):
                if name.endswith('.img') and self.ppms[0] in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)

                    try:
                        ch1 = data.reshape((400, 400))
                        ch1 = np.flipud(ch1)
                        patch1=self.getPatch(ch1,patchCoor)
                        patch1 = self.pad(patch1)
                    except ValueError:
                        print(str(id)+'is not 400 by 400, skipped to next...')
                        fid.close()
                if name.endswith('.img') and self.ppms[1] in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    try:
                        ch2 = data.reshape((400, 400))
                        ch2 = np.flipud(ch2)
                        patch2 = self.getPatch(ch2, patchCoor)
                        patch2 = self.pad(patch2)
                    except ValueError:
                        print(str(id)+'is not 400 by 400, skipped to next...')
                        fid.close()
                if name.endswith('.img') and self.ppms[2] in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    try:
                        ch3 = data.reshape((400, 400))
                        ch3 = np.flipud(ch3)
                        patch3 = self.getPatch(ch3, patchCoor)
                        patch3 = self.pad(patch3)
                    except ValueError:
                        print(str(id)+'is not 400 by 400, skipped to next...')
                        fid.close()
        elif id in self.kki_list:
            for name in os.listdir(dir):
                if name.endswith('.img'):
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    apt_cube = data.reshape((15,256,256))
                    fid.close()
                    apt_img = apt_cube[random.randint(6,9),:,:]
                    # get the middle three slice from kki apt cube randomly
        else:
            print(str(id)+' not found')
        df_index = int(id)-1
        label = self.dataframe.loc[df_index,'IDH']
        age = self.dataframe.loc[df_index,'AGE']
        #import matplotlib; matplotlib.pyplot.imshow(mydata)
        if self.switch ==1:
            three_channel_img = np.stack([ch1,ch2,ch3],axis = 0)
            pil_img = np.stack([ch1, ch2, ch3], axis=2)
            three_channel_patch=np.stack([patch1,patch2,patch3],axis=0)
            pil_patch = np.stack([patch1,patch2,patch3],axis=2)

        else:

            three_channel_img = np.repeat(apt_img[np.newaxis,:, :], 3, axis=0)
            pil_img = np.repeat(apt_img[:, :,np.newaxis], 3, axis=2)

        if self.transform:

            three_channel_img=self.applyTransform(pil_img)
            #three_channel_patch=torch.tensor(three_channel_patch)
            three_channel_patch = self.applyTransform(pil_patch)
            #three_channel_patch = self.applyTransform(pil_patch)
        else:
            three_channel_img = torch.from_numpy(three_channel_img)
            three_channel_patch= torch.from_numpy(three_channel_patch)
        age = torch.tensor(float(age))
        label = torch.tensor(int(label)-1)
        sample = {'image':three_channel_img,
                  'patch':three_channel_patch,
                  'age':age,'label':label,'id':id}

        return sample


    def applyTransform(self,array):
        array= array.clip(min=-5,max=5)+5
        array*= (255.0 / array.max())
        array = array.astype(np.uint8)
        # the type casting is very important, because PIL won't work on floating point
        pil_img_obj = Image.fromarray(array, 'RGB')
        pil_img_obj = self.transform(pil_img_obj)
        # already tensor, gotta remove the below conversion lines
        #pil_ary = pil_img_obj.numpy()
        #torchImg = torch.from_numpy(pil_ary)
        return pil_img_obj
        # fix: return value changed from torchImg to pil_img_obj

##import matplotlib; matplotlib.pyplot.imshow(mydata)