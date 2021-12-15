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


class Kadis700kFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.imgname = []
        self.mos = []
        self.mat_file = os.path.join(self.root, 'kadis700k.mat')
        datainfo = scipy.io.loadmat(self.mat_file)
        image_number = len(datainfo['ref_img_name'])
        for i in range(0, image_number):  # image_number
            self.imgname.append(datainfo['dis_img_name'][i][0][0])
            mos = float(datainfo['label'][i][0])
            mos = np.array(mos)
            mos = mos.astype(np.float32)
            self.mos.append(mos)

        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, 'dist_imgs', self.imgname[item * 5]), self.mos[item * 5]))
            sample.append((os.path.join(self.root, 'dist_imgs', self.imgname[item * 5 + 1]), self.mos[item * 5 + 1]))
            sample.append((os.path.join(self.root, 'dist_imgs', self.imgname[item * 5 + 2]), self.mos[item * 5 + 2]))
            sample.append((os.path.join(self.root, 'dist_imgs', self.imgname[item * 5 + 3]), self.mos[item * 5 + 3]))
            sample.append((os.path.join(self.root, 'dist_imgs', self.imgname[item * 5 + 4]), self.mos[item * 5 + 4]))
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
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

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
    trainset = Kadis700kFolder(root=Kadid10kroot, loader=default_loader, index=train_index)
    testset = Kadis700kFolder(root=Kadid10kroot, loader=default_loader, index=test_index)