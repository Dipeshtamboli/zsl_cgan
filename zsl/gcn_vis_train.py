from torchvision.models import resnet152
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
from old_16_frames import VideoDataset

import torch.nn as nn
import torch.nn.functional as F
import torch
from time import time
from zsl_models import *
from gcn_model import GCN

start_time = time()
parser = argparse.ArgumentParser(description='Video action recogniton using GCN and semantics')
parser.add_argument('--logfile_name', type=str, default="gcn_vis_10",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
args = parser.parse_args()

num_epochs = 50
b1=0.5
b2=0.999
seen_cls = 51
unseen_cls = 10
n_classes = seen_cls + unseen_cls

save_epoch = 10
# final_total_class = 51
latent_dim = 300+100
lr = 1e-3



gpu_id = str(args.gpu)
log_name = args.logfile_name
writer = SummaryWriter(log_dir=log_name)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
save_dir = os.path.join(save_dir_root, log_name)
modelName = 'GCN-SEM-VIS' # Options: C3D or R2Plus1D or R3D
saveName = modelName

unseen_att = np.load("../cgan_training/unseen_semantic_50.npy")[:unseen_cls]
att = np.load("../cgan_training/seen_semantic_51.npy")
att = np.concatenate((att,unseen_att),0)
att = torch.tensor(att).cuda()
att = att.type(torch.FloatTensor).cuda()

cls_criterion = nn.CrossEntropyLoss().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(device)
# cls_criterion.to(device)
# model.to(device)
# train_dataloader = DataLoader(VideoDataset(dataset='ucf101', all_data=True, split='train',clip_len=16), batch_size=100, shuffle=True, num_workers=4)




cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
adj = torch.zeros((n_classes,n_classes)).cuda()
for i in range(n_classes):
    for j in range(n_classes):
        adj[i][j] = cos(att[i].unsqueeze(0),att[j].unsqueeze(0))

gcn_model = GCN(nfeat=att.shape[1],
            nhid=128,
            nclass=n_classes,
            dropout=0.5).cuda()
gcn_model.to('cuda')

visual_model = Visual_FC()
visual_model.to('cuda')

sem_model = Semantic_FC()
sem_model.to('cuda')

sem_graph_model = Semantic_and_Graph_FC()
sem_graph_model.to('cuda')

classifier = Classifier(n_classes)
classifier.to('cuda')

print('visual_model: Total params: %.2fM' % (sum(p.numel() for p in visual_model.parameters()) / 1000000.0))
print('sem_model: Total params: %.2fK' % (sum(p.numel() for p in sem_model.parameters()) / 1000.0))
print('classifier: Total params: %.2fK' % (sum(p.numel() for p in classifier.parameters()) / 1000.0))
print('gcn_model: Total params: %.2fK' % (sum(p.numel() for p in gcn_model.parameters()) / 1000.0))
print('sem_graph_model: Total params: %.2fK' % (sum(p.numel() for p in sem_graph_model.parameters()) / 1000.0))
optimizer = torch.optim.Adam(list(visual_model.parameters())+list(sem_graph_model.parameters())+list(sem_model.parameters())+list(gcn_model.parameters())+list(classifier.parameters()), lr=1e-5)

# /home/SharedData/fabio/zsl_cgan/cgan_training/gn_feats/classes_51_generated.npy
lstm_feats = np.load("../cgan_training/gn_feats/classes_51_add_10_generated.npy")
lstm_features = torch.tensor(lstm_feats[:,:-1])
lstm_labels = torch.tensor(lstm_feats[:,-1])
train_true = data_utils.TensorDataset(lstm_features, lstm_labels)
train_dataloader = data_utils.DataLoader(train_true, batch_size=100, shuffle=True)

trainval_loaders = {'train': train_dataloader}
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train']}
# exit()
for epoch in range(num_epochs):
    
    lab_list=[]
    pred_list=[]
    for phase in ['train']:
        # start_time = timeit.default_timer()

        # reset the running loss and corrects
        running_loss = 0.0
        loss_cel = 0.0
        loss_triplet = 0.0
        running_corrects = 0.0

        # set model to train() or eval() mode depending on whether it is trained
        # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        if phase == 'train':
            # scheduler.step() is to be called once every epoch during training
            # scheduler.step()
            visual_model.train()
            sem_model.train()
            gcn_model.train()
            classifier.train()
        else:
            visual_model.eval()
            sem_model.eval()
            gcn_model.eval()
            classifier.eval()

        for inputs, labels in (trainval_loaders[phase]):
            # move inputs and labels to the device the training is taking place on
            # inputs = inputs.permute(0,2,1,3,4)
            image_sequences = Variable(inputs.to(device), requires_grad=True)
            labels = Variable(labels.to(device), requires_grad=False)                

            # inputs = Variable(inputs, requires_grad=True).to(device)
            # labels = Variable(labels).to(device)
            optimizer.zero_grad()
            # model.lstm.reset_hidden_state()

            if phase == 'train':
                vis_feats_256 = visual_model(image_sequences)
                # sem_out_128 = sem_model(att.type(torch.FloatTensor).cuda()[labels])
                sem_out_128 = sem_model(att[labels.type(torch.LongTensor)])
                gcn_out_128 = gcn_model(att,adj)[labels.type(torch.LongTensor)]
                semantic_256_feats = torch.cat((sem_out_128,gcn_out_128), 1)
                modified_semantic_256_feats = sem_graph_model(semantic_256_feats)
                # pdb.set_trace()
                predictions = classifier(torch.cat((vis_feats_256,modified_semantic_256_feats), 1))
            else:
                with torch.no_grad():
                    predictions, lstm_out = visual_model(image_sequences)

            # loss = criterion(outputs, labels)
            classification_loss = cls_criterion(predictions, labels.type(torch.LongTensor).cuda())
            trip_loss = triplet_loss(semantic_256_feats,vis_feats_256,vis_feats_256[torch.randperm(len(vis_feats_256))])
            # pdb.set_trace()
            loss =  classification_loss + trip_loss
            acc = 100 * (predictions.detach().argmax(1) == labels.type(torch.LongTensor).cuda()).cpu().numpy().mean()
            probs = nn.Softmax(dim=1)(predictions)
            preds = torch.max(probs, 1)[1]

            lab_list += labels.cpu().numpy().tolist()
            pred_list += preds.cpu().numpy().tolist()
            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            loss_cel += classification_loss.item() * inputs.size(0)
            loss_triplet += trip_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.type(torch.LongTensor).cuda().data)

        # conf_mat = confusion_matrix(lab_list, pred_list)
        # np.save("{}.npy".format(os.path.join(save_dir, saveName + '_epoch-' + str(epoch))+'_'+ phase), conf_mat)
        # fig = plt.figure()
        # plt.imshow(conf_mat)
        # writer.add_figure('conf_mat_'+phase, fig, epoch)
        epoch_loss = running_loss / trainval_sizes[phase]
        epoch_cel_loss = loss_cel / trainval_sizes[phase]
        epoch_trip_loss = loss_triplet / trainval_sizes[phase]
        epoch_acc = running_corrects.double() / trainval_sizes[phase]

        if phase == 'train':
            writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/train_cel_loss_epoch', epoch_cel_loss, epoch)
            writer.add_scalar('data/train_trip_loss_epoch', epoch_trip_loss, epoch)
            writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        else:
            writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, num_epochs, epoch_loss, epoch_acc))

        # stop_time = timeit.default_timer()
        # print("Execution time: " + str(stop_time - start_time) + "\n")
    save_path = os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')
    if epoch % save_epoch == (save_epoch - 1):
        torch.save({
            'epoch': epoch + 1,
            'visual_state_dict': visual_model.state_dict(),
            'sem_graph_state_dict': sem_graph_model.state_dict(),
            'sem_state_dict': sem_model.state_dict(),
            'gcn_state_dict': gcn_model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'opt_dict': optimizer.state_dict(),
        }, save_path)
        print("Save model at {}\n".format(save_path))

print(f'Execution time: {(time()- start_time)//3600} hrs \
  {(time()- start_time)%3600//60} min {int((time()- start_time)%60)} sec')

# for inputs, labels in (train_dataloader):
#     inputs = inputs.permute(0,2,1,3,4)
#     image_sequences = Variable(inputs.to("cuda"), requires_grad=True)        
#     out = model(image_sequences)
#     pdb.set_trace()