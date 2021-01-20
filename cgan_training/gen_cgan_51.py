from tensorboardX import SummaryWriter
import torch.utils.data as data_utils
import scipy.io as sio
import pdb
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
n_epochs = 200
b1=0.5
b2=0.999
n_classes = 51
unseen_cls = 10
final_total_class = 101
latent_dim = 300+100
lr = 1e-3
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(final_total_class, final_total_class)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + final_total_class, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 2048),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        return img



cuda = True if torch.cuda.is_available() else False
generator = Generator()

checkpoint = torch.load("saved_models/classes_51_epoch-399.pth.tar",
               map_location=lambda storage, loc: storage)

generator.load_state_dict(checkpoint['gen_state_dict'])

generator.cuda()

unseen_att = np.load("unseen_semantic_50.npy")[:unseen_cls]
att = np.load("seen_semantic_51.npy")
att = np.concatenate((att,unseen_att),0)
att = torch.tensor(att).cuda()

# split_1 = sio.loadmat("/home/SharedData/fabio/cgan/hmdb_i3d/split_1/att_splits.mat")
# att = split_1["att"]
# att = torch.tensor(att).cuda()
# att = torch.transpose(att,1,0)
# att = att[:n_classes] #first 25 classes

LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# batch_size = n_classes

# gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
gen_labels = Variable(LongTensor(np.arange(n_classes+ unseen_cls)))
gen_labels = torch.cat(100*[gen_labels])
z = Variable(FloatTensor(np.random.normal(0, 1, (len(gen_labels), latent_dim))))
z[:,:300] = (att[gen_labels])

# Generate a batch of images
gen_imgs = generator(z, gen_labels)
# pdb.set_trace()
gen_feats_labs = torch.cat((gen_imgs,gen_labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)
np.save("classes_51_add_10_generated.npy", gen_feats_labs.cpu().detach().numpy())