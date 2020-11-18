from tensorboardX import SummaryWriter
import torch.utils.data as data_utils
import scipy.io as sio
import pdb
import argparse
import os
import numpy as np
import math
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

log_name = "classes_51"
writer = SummaryWriter(log_dir="logs/{}".format(log_name))

n_epochs = 400
b1=0.5
b2=0.999
# n_classes = 25+16+10
n_classes = 51
use_pretrained = False
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(final_total_class, final_total_class)

        self.model = nn.Sequential(
            nn.Linear(final_total_class + 2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


if use_pretrained:
    print("using pretrained model")
    checkpoint = torch.load("/home/SharedData/fabio/cgan/saved_models/classes_0-24+25-40(2)_model_epoch-399.pth.tar",
                   map_location=lambda storage, loc: storage)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    discriminator.load_state_dict(checkpoint['dis_state_dict'])
    generator.train()
    discriminator.train()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# label_to_id = json.load(open("all_seen_unseen_labs.json"))
# split_1 = sio.loadmat("att_splits.mat")
# att = split_1["att"]
# att = torch.tensor(att).cuda()
# att = torch.transpose(att,1,0)
# att = att[:n_classes] #first 25 classes

att = np.load("seen_semantic_51.npy")
att = torch.tensor(att).cuda()
# pdb.set_trace()
lstm_feats = np.load("../npy_files/lstm_feats_51_classes_2048d.npy")
# gen_feats = np.load("data/gen_feats_0-24+25-40(2).npy") #features of first 25 classes
# # lstm_feats = np.load("data/lstm_feats_0-24_classes_2696x2049d.npy") #features of first 25 classes
# lstm_feats = np.load("data/lstm_feats_41-50_classes_2696x2049d.npy") #features of first 25 classes
# lstm_feats = np.concatenate((gen_feats, lstm_feats))
lstm_features = torch.tensor(lstm_feats[:,:-1])
lstm_labels = torch.tensor(lstm_feats[:,-1])

train_true = data_utils.TensorDataset(lstm_features, lstm_labels)
dataloader = data_utils.DataLoader(train_true, batch_size=100, shuffle=True)

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
        z[:,:300] = (att[gen_labels])

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        # pdb.set_trace()
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        writer.add_scalar('dis_loss', d_loss.item(), epoch)
        writer.add_scalar('gen_loss', g_loss.item(), epoch)
        writer.add_scalar('total_loss', d_loss.item()+g_loss.item(), epoch)


        batches_done = epoch * len(dataloader) + i
        # if batches_done % sample_interval == 0:
        #     sample_image(n_row=10, batches_done=batches_done)
save_name = "saved_models/{}_epoch-{}.pth.tar".format(log_name,str(epoch))
torch.save({
    'epoch': epoch + 1,
    'gen_state_dict': generator.state_dict(),
    'dis_state_dict': discriminator.state_dict(),
    'D_opt_dict': optimizer_D.state_dict(),
    'G_opt_dict': optimizer_G.state_dict(),
}, os.path.join(save_name))
print("model is saved at: {}".format(save_name))