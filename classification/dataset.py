import os
import numpy as np
import cv2
import torch
from PIL import Image

# dataloader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.list_input = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.list_input)

    def __getitem__(self, index):
        input_name = self.list_input[index]
        input = cv2.imread(os.path.join(self.data_dir, input_name))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # input = cv2.resize(input, dsize=(640,640))
        # input = input/255.0
        input = Image.fromarray(input)

        split_input_name = input_name.split('_')
        class_name = split_input_name[1]

        if class_name == 'noise':
            label = 0
        elif class_name == 'book':
            label = 1
        elif class_name == 'chair':
            label = 2
        elif class_name == 'desk':
            label = 3
        elif class_name == 'lecturestand':
            label = 4
        elif class_name == 'hammer':
            label = 5

        if self.transform:
            input = self.transform(input)

        return input, label

# # transform
# class ToTensor(object):
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         label = np.asarray([label])
#         label = label.astype(np.float32)
#         input = input.transpose((2, 0, 1)).astype(np.float32)

#         data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

#         return data

# class Normalization(object):
#     def __init__(self, mean=0.5, std=0.5):
#         self.mean = mean
#         self.std = std

#     def __call__(self, data):
#         label, input = data['label'], data['input']
#         input = (input - self.mean) / self.std
#         data = {'label': label, 'input': input}

#         return data