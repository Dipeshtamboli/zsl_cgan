import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import pdb
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import time
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import requests
from old_16_frames import VideoDataset
from lstm_models import *

parser = argparse.ArgumentParser(description='Video action recogniton testing(feature extraction to a npy file)')
parser.add_argument('--logfile_name', type=str, default="LSTM_part2",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
args = parser.parse_args()

gpu_id = str(args.gpu)
log_name = args.logfile_name
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101
num_classes=51

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
save_dir = os.path.join(save_dir_root, 'run', log_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

model = ConvLSTM(
    num_classes=num_classes,
    latent_dim=512,
    lstm_layers=1,
    hidden_dim=1024,
    bidirectional=True,
    attention=True,
)
model = model.to(device)
# /home/SharedData/fabio/c3d_codes/run/LSTM_part2/Bi-LSTM-ucf101_epoch-199.pth.tar
# /home/SharedData/fabio/zsl_cgan/seen51_training/run/bi-lstm_seen51_training/Bi-LSTM-ucf101_epoch-199.pth.tar
# /home/SharedData/fabio/zsl_cgan/seen51_training/run/LSTM_part2/Bi-LSTM-ucf101_epoch-199.pth.tar
load_epoch = 199
# checkpoint = torch.load(os.path.join(save_dir, saveName + '_epoch-' + f"{load_epoch}" + '.pth.tar'),
#                map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
load_Path = "/home/SharedData/fabio/zsl_cgan/seen51_training/run/bi-lstm_seen51_training/Bi-LSTM-ucf101_epoch-199.pth.tar"
checkpoint = torch.load(load_Path, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU

print("Initializing weights from: {}...".format(
    os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))

model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['opt_dict'])
print('Training model on {} dataset...'.format(dataset))
# pdb.set_trace()

train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test_unseen',clip_len=16), batch_size=100, shuffle=False, num_workers=4)
# all_dataloader = DataLoader(VideoDataset(dataset=dataset, all_data=True, split='train',clip_len=16), batch_size=100, shuffle=False, num_workers=4)
all_dataloader = train_dataloader


model.eval()
lab_list = []
pred_list = []
running_corrects = 0.0

lstm_feats = np.zeros((1,2049))
for inputs, labels in (all_dataloader):
    # move inputs and labels to the device the training is taking place on
    inputs = inputs.permute(0,2,1,3,4)
    image_sequences = Variable(inputs.to(device), requires_grad=True)
    labels = Variable(labels.to(device), requires_grad=False)                

    # optimizer.zero_grad()
    model.lstm.reset_hidden_state()

    with torch.no_grad():
        predictions, lstm_out = model(image_sequences)

        # pdb.set_trace()
        # lstm_out = torch.cat(lstm_out, labels, axis=1)
        # labels = labels.unsqueeze(1)
        lstm_out = torch.cat((lstm_out, labels.type(torch.cuda.FloatTensor).unsqueeze(1)),dim= 1)
    # pdb.set_trace()
    # np.append(lstm_feats, lstm_out)
    lstm_feats = np.append(lstm_feats, lstm_out.cpu(), axis=0)
    # loss = criterion(outputs, labels)
    # loss = cls_criterion(predictions, labels)
    acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
    probs = nn.Softmax(dim=1)(predictions)
    preds = torch.max(probs, 1)[1]

    lab_list += labels.cpu().numpy().tolist()
    pred_list += preds.cpu().numpy().tolist()

    running_corrects += torch.sum(preds == labels.data)

conf_mat = confusion_matrix(lab_list, pred_list)
# np.save("{}.npy".format(os.path.join(save_dir, saveName + '_epoch-' + str(epoch))+'_'+ phase), conf_mat)
# fig = plt.figure()
# plt.imshow(conf_mat)
# writer.add_figure('conf_mat_'+phase, fig, epoch)
epoch_acc = running_corrects.double() / len(all_dataloader)

print("Acc: {}".format(epoch_acc))
# stop_time = timeit.default_timer()
# print("Execution time: " + str(stop_time - start_time) + "\n")
np.save(f"lstm_feats_50_unseen_classes_2048d.npy", lstm_feats[1:])
print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")
