
# -*- coding: utf-8 -*-
"""Tsne.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/11iqaTgnE4Hcxo0BYwALq2io9TFaSTxfO
"""

import os

import torch
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
from nets import *
from video_data_loader import video_dataset, old_video_dataset

model_path = "run/pipeline_set1/Bi-LSTM-ucf101_increment_epoch-149.pth.tar"
att = np.load("../npy_files/seen_semantic_51.npy")
att = torch.tensor(att).cuda()    


semantic_dim = 300
noise_dim = 1024

total_classes = 40
all_classes = np.arange(total_classes)


model = ConvLSTM(
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

generator = Modified_Generator(semantic_dim, noise_dim)

checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
model.load_state_dict(checkpoint['extractor_state_dict'])
generator.load_state_dict(checkpoint['generator_state_dict'])

model = model.cuda()
generator = generator.cuda()

train_dataset = old_video_dataset(train = True, classes = all_classes[:total_classes])
train_dataloader = DataLoader(train_dataset, batch_size = 100, shuffle = False, num_workers = 4)

for i, (_, inputs, labels) in enumerate(train_dataloader):
    inputs = inputs.permute(0,2,1,3,4)
    inputs = Variable(inputs.cuda())
    #inputs = inputs.contiguous().view(inputs.size(0), -1)
    inputs = inputs.cuda()
    labels = labels.cuda()
    noise = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (100, noise_dim)))).cuda()
    semantic_true = att[labels].cuda()

    convlstm_feats = model(inputs)
    convlstm_feats = model(inputs).contiguous().view(convlstm_feats.size(0), -1)
    generator_feats = generator(semantic_true.float(), noise)
    
    if (i == 0):
        convlstm_feat = convlstm_feats
        #convlstm_feat = inputs
        generator_feat = generator_feats

    else:
        convlstm_feat = torch.cat((convlstm_feat, convlstm_feats), 0)
        #convlstm_feat = torch.cat((convlstm_feat, inputs), 0)
        generator_feat = torch.cat((generator_feat, generator_feats), 0)


#mask_feat_path = "/home/SharedData/fabio/zsl_cgan/new_model/masked_feat_1629x1280.npy"
#non_mask_feat_path = "/home/SharedData/fabio/zsl_cgan/new_model/non_masked_feat_1994x1280.npy"

#mask_feat = np.load(mask_feat_path)
#non_mask_feat = np.load(non_mask_feat_path)

convlstm_feat = convlstm_feat.squeeze_(0)
generator_feat = generator_feat.squeeze_(0)

convlstm_feat = convlstm_feat.cpu()
generator_feat = generator_feat.cpu()

convlstm_feat = convlstm_feat.detach().numpy()
generator_feat = generator_feat.detach().numpy()

print("Conv LSTM feature shape {}".format(convlstm_feat.shape))
print("Generator feature shape {}".format(generator_feat.shape))

all_features = np.concatenate((convlstm_feat, generator_feat), axis = 0)
#all_features = convlstm_feat
dataset_label = np.zeros((all_features.shape[0],1))
dataset_label[convlstm_feat.shape[0]:] = 1

#for i in range(total_classes):
    #dataset_label[10*i:10*i+10] = i

start_time = time.time()

tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)

plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], legend='full', palette=palette)
#plt.savefig("2048_tsne.png")
plt.savefig("dataset_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))