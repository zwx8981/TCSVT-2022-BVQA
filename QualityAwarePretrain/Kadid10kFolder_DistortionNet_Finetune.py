import torch.utils.data as data

from PIL import Image

import os
import os.path
# import math
import scipy.io
import numpy as np
import random
import csv


def getFileName(path, suffix):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0, num):
        name = '%s%s%s' % ('img', str(index), '.bmp')
        filename.append(os.path.join(path, name))
        index = index + 1
    return filename


class Kadid10kFolder_DistortionNet_Finetune(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.imgname = []
        self.mos = []
        self.d_type = []
        self.d_level = []
        self.mat_file = os.path.join(self.root, 'kadid10k.mat')
        datainfo = scipy.io.loadmat(self.mat_file)
        image_number = len(datainfo['ref_img_name'])
        for i in range(0, image_number):  # image_number
            self.imgname.append(datainfo['dis_img_name'][i][0][0])
            mos = float(datainfo['label'][i][0])
            mos = np.array(mos)
            mos = mos.astype(np.float32)
            self.mos.append(mos)

            d_type = float(datainfo['d_type'][i][0])
            d_type = np.array(d_type)
            d_type = d_type.astype(np.int64)
            self.d_type.append(d_type)

            d_level = float(datainfo['d_level'][i][0])
            d_level = np.array(d_level)
            d_level = d_level.astype(np.int64)
            self.d_level.append(d_level)

        sample = []
        for i, item in enumerate(index):
            sample.append(item)
        self.samples = sample
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        ref_idx = self.samples[index]

        image_name1 = os.path.join(self.root, 'images', self.imgname[ref_idx * 5])
        image_name2 = os.path.join(self.root, 'images', self.imgname[ref_idx * 5 + 1])
        image_name3 = os.path.join(self.root, 'images', self.imgname[ref_idx * 5 + 2])
        image_name4 = os.path.join(self.root, 'images', self.imgname[ref_idx * 5 + 3])
        image_name5 = os.path.join(self.root, 'images', self.imgname[ref_idx * 5 + 4])
        I1 = self.loader(image_name1)
        I2 = self.loader(image_name2)
        I3 = self.loader(image_name3)
        I4 = self.loader(image_name4)
        I5 = self.loader(image_name5)
        if self.transform is not None:
            I1 = self.transform(I1)
            I2 = self.transform(I2)
            I3 = self.transform(I3)
            I4 = self.transform(I4)
            I5 = self.transform(I5)

        I1_D = self.d_type[ref_idx * 5] - 1
        I2_D = self.d_type[ref_idx * 5 + 1] - 1
        I3_D = self.d_type[ref_idx * 5 + 2] - 1
        I4_D = self.d_type[ref_idx * 5 + 3] - 1
        I5_D = self.d_type[ref_idx * 5 + 4] - 1

        I1_DL = 6 - self.d_level[ref_idx * 5]
        I2_DL = 6 - self.d_level[ref_idx * 5 + 1]
        I3_DL = 6 - self.d_level[ref_idx * 5 + 2]
        I4_DL = 6 - self.d_level[ref_idx * 5 + 3]
        I5_DL = 6 - self.d_level[ref_idx * 5 + 4]

        I1_M = self.mos[ref_idx * 5]
        I2_M = self.mos[ref_idx * 5 + 1]
        I3_M = self.mos[ref_idx * 5 + 2]
        I4_M = self.mos[ref_idx * 5 + 3]
        I5_M = self.mos[ref_idx * 5 + 4]

        # sample = []
        sample = {'I1': I1, 'I1_D': I1_D, 'I1_DL': I1_DL, 'I1_M': I1_M,
                  'I2': I2, 'I2_D': I2_D, 'I2_DL': I2_DL, 'I2_M': I2_M,
                  'I3': I3, 'I3_D': I3_D, 'I3_DL': I3_DL, 'I3_M': I3_M,
                  'I4': I4, 'I4_D': I4_D, 'I4_DL': I4_DL, 'I4_M': I4_M,
                  'I5': I5, 'I5_D': I5_D, 'I5_DL': I5_DL, 'I5_M': I5_M
                 }

        return sample

    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


if __name__ == '__main__':
    Kadid10kroot = '/mnt/sda2/New/kadis700k/kadis700k'
    index = list(range(0, 129109))
    random.shuffle(index)
    train_index = index[0:round(0.8 * 129109)]
    test_index = index[round(0.8 * 129109):129109]
    trainset = Kadid10kFolder_DistortionNet_Finetune(root=Kadid10kroot, loader=default_loader, index=train_index)
    testset = Kadid10kFolder_DistortionNet_Finetune(root=Kadid10kroot, loader=default_loader, index=test_index)