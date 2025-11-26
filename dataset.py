from torch.utils.data import Dataset
import cv2
import numpy as np
from torch.utils.tensorboard.summary import image
import os

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # height * width
    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape((1, *xray.shape))     # 1 * height * width
    return xray

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape((1, *mask.shape))
    return mask

class knee_dataset(Dataset):
    def __init__(self, df, has_mask=True):
        self.df = df
        self.has_mask = has_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = read_xray(self.df['xrays'].iloc[index])
        result = {'image': image}

        if self.has_mask:
            mask = read_mask(self.df['masks'].iloc[index])
            result['mask'] = mask
        else:
            result['name'] = os.path.basename(self.df['xrays'].iloc[index])

        return result
'''
    def __getitem__(self, index):
        image = read_xray(self.df['xrays'].iloc[index])
        mask = read_mask(self.df['masks'].iloc[index])
        

        result = {
            'image': image,
            'mask': mask
        }

        return result
'''