from old_16_frames import VideoDataset
import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class video_dataset(Dataset):
    def __init__(self, train = True, classes = range(10)):
        self.train = train
        self.classes = classes

        if self.train == True:
            split = 'data_1_train'
        else:
            split = 'data_1_test'

        self.dataset = VideoDataset(dataset='ucf101', split= split, clip_len=16)

        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.dataset)):
                images, labels = self.dataset[i]
                
                if int(labels) in self.classes:
                    train_data.append(images)
                    train_labels.append(int(labels))
					
            self.train_data = train_data
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.dataset)):
                images, labels = self.dataset[i]
                    
                if int(labels) in self.classes:
                    test_data.append(images)
                    test_labels.append(int(labels))
					
            self.test_data = test_data
            self.test_labels = test_labels


    def __getitem__(self, index):
        if self.train:
            image = self.train_data[index]
            image = torch.FloatTensor(image)
            target = self.train_labels[index]
        
        else:
            image, target = self.test_data[index], self.test_labels[index]

        image = torch.FloatTensor(image)	
        return index, image, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)