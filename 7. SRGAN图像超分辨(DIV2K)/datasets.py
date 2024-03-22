import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import csv
import random
from PIL import Image
from imresize import imresize
import numpy as np 

class SRDataset(Dataset):
    
    def __init__(self, split, config):

        self.split = split
        self.csv_folder = config.csv_folder
        self.HR_data_folder = config.HR_data_folder
        self.LR_data_folder = config.LR_data_folder
        self.crop_size = int(config.crop_size)
        self.scaling_factor = int(config.scaling_factor)
        self.patch_num = 100

        assert self.split in {'train', 'valid', 'Set5', 'Set14', 'B100', 'Urban100'}

        self.HR_images = []
        self.LR_images = []
        with open(os.path.join(self.csv_folder, self.split + '_images.csv'), 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.HR_images.append(line[0])
                self.LR_images.append(line[1])

    def __getitem__(self, i):

        if self.split != 'train':
            hr_image = Image.open(os.path.join(self.HR_data_folder, self.HR_images[i]), mode='r')
            lr_image = Image.open(os.path.join(self.LR_data_folder, self.LR_images[i]), mode='r')
            hr_image = hr_image.convert('RGB')
            lr_image = lr_image.convert('RGB')
            
            if lr_image.width * self.scaling_factor != hr_image.width or lr_image.height * self.scaling_factor != hr_image.height:
                x_remainder = hr_image.width % self.scaling_factor
                y_remainder = hr_image.height % self.scaling_factor
                left = x_remainder // 2
                top = y_remainder // 2
                right = left + (hr_image.width - x_remainder)
                bottom = top + (hr_image.height - y_remainder)
                hr_image = hr_image.crop((left, top, right, bottom))
                hr_image = np.asarray(hr_image)
                lr_image = imresize(hr_image, scalar_scale=1.0/self.scaling_factor)
                lr_image = Image.fromarray(np.uint8(lr_image))
                hr_image = Image.fromarray(np.uint8(hr_image))
            
            lr_image = transforms.functional.to_tensor(lr_image)
            hr_image = transforms.functional.to_tensor(hr_image)

            return lr_image, hr_image
        
        if self.split == 'train':
            i = i // self.patch_num
        
        hr_image = Image.open(os.path.join(self.HR_data_folder, self.HR_images[i]), mode='r')
        lr_image = Image.open(os.path.join(self.LR_data_folder, self.LR_images[i]), mode='r')
        hr_image = hr_image.convert('RGB')
        lr_image = lr_image.convert('RGB')

        if lr_image.width * self.scaling_factor != hr_image.width or lr_image.height * self.scaling_factor != hr_image.height:
            x_remainder = hr_image.width % self.scaling_factor
            y_remainder = hr_image.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (hr_image.width - x_remainder)
            bottom = top + (hr_image.height - y_remainder)
            hr_image = hr_image.crop((left, top, right, bottom))
            hr_image = np.asarray(hr_image)
            lr_image = imresize(hr_image, scalar_scale=1.0/self.scaling_factor)
            lr_image = Image.fromarray(np.uint8(lr_image))
            hr_image = Image.fromarray(np.uint8(hr_image))

        lr_image, hr_image = self._sample_patch(lr_image, hr_image)
        lr_image, hr_image = self._augment(lr_image, hr_image)
        lr_image = transforms.functional.to_tensor(lr_image)
        hr_image = transforms.functional.to_tensor(hr_image)

        return lr_image, hr_image

    def _sample_patch(self, lr_image, hr_image):

        if self.split == 'train':
            # sample patch while training
            lr_crop_size = self.crop_size // self.scaling_factor
            left = random.randint(2, lr_image.width - lr_crop_size - 2)
            top = random.randint(2, lr_image.height - lr_crop_size - 2)
            right = left + lr_crop_size
            bottom = top + lr_crop_size
            lr_image = lr_image.crop((left, top, right, bottom))
            hr_image = hr_image.crop((left * self.scaling_factor, top * self.scaling_factor, right * self.scaling_factor, bottom * self.scaling_factor))

        return lr_image, hr_image

    def _augment(self, lr_image, hr_image):

        if self.split == 'train':
            # augmentation while training
            if random.random() < 0.5:
                lr_image = lr_image.transpose(Image.FLIP_LEFT_RIGHT)
                hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                lr_image = lr_image.transpose(Image.FLIP_TOP_BOTTOM)
                hr_image = hr_image.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                lr_image = lr_image.rotate(90)
                hr_image = hr_image.rotate(90)

        return lr_image, hr_image

    def __len__(self):

        if self.split == 'train':
            return len(self.HR_images) * self.patch_num
        else:
            return len(self.HR_images)
