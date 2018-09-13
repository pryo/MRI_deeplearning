import os
import random

from PIL import Image
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader

class APTDataset(Dataset):
    """load the apt.img only"""
    def __init__(self,root_dir,gz_list,kki_list,truth_path='idh.xlsx',transform=None, switch = 0):
        # switch =0 for all
        #1 for gz
        # 2 for kki
        self.switch = switch
        self.root_dir = root_dir
        self.transform = transform
        self.gz_list = gz_list
        self.kki_list = kki_list

        self.dataframe = pd.read_excel(truth_path)
        self.odd_list = [10,49,73,74,75,76,8,97,100]

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
        if id in self.gz_list:

            for name in os.listdir(dir):
                if name.endswith('.img') and 'aptw' in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    try:
                        ch1 = data.reshape((400, 400))
                    except ValueError:
                        print(str(id)+'is not 400 by 400, skipped to next...')
                        fid.close()
                if name.endswith('.img') and '1.5ppm' in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    try:
                        ch2 = data.reshape((400, 400))
                    except ValueError:
                        print(str(id)+'is not 400 by 400, skipped to next...')
                        fid.close()
                if name.endswith('.img') and '2.5ppm' in name:
                    apt_img_path = os.path.join(dir,name)
                    dtype = np.dtype('float32')  # big-endian unsigned integer (16bit)
                    # Reading.
                    fid = open(apt_img_path, 'rb')
                    data = np.fromfile(fid, dtype)
                    try:
                        ch3 = data.reshape((400, 400))
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
        if self.switch ==1:
            three_channel_img = np.stack([ch1,ch2,ch3],axis = 0)
            pil_img = np.stack([ch1, ch2, ch3], axis=2)
        else:

            three_channel_img = np.repeat(apt_img[np.newaxis,:, :], 3, axis=0)
            pil_img = np.repeat(apt_img[:, :,np.newaxis], 3, axis=2)

        if self.transform:
            pil_img = pil_img.clip(min=-5,max=5)+5
            pil_img *= (255.0 / pil_img.max())
            pil_img_obj = Image.fromarray(pil_img, 'RGB')
            pil_img_obj=self.transform(pil_img_obj)
            pil_ary=pil_img_obj.numpy()
            three_channel_img= torch.from_numpy(pil_ary)
            #three_channel_img=np.asarray(pil_img_obj)
#            three_channel_img=pil_ary.transpose((2, 0, 1))

        else:
            three_channel_img = torch.from_numpy(three_channel_img)
        age = torch.tensor(float(age))
        label = torch.tensor(int(label)-1)
        sample = {'image':three_channel_img,
                  'age':age,'label':label}

        return sample


