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

def send_dipesh(send_string):
#    chatwork.send_message(log_id, send_string)
    headers = {
        'Content-type': 'application/json',
    }
    data = '{}{}{}'.format("{\"text\":\"",send_string,"\"}")
    # response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=data)

send_dipesh("test timestamp: {}".format(datetime.now()))
send_dipesh("python file name: "+os.path.abspath(__file__))
send_dipesh("--- UCF code started ---")
# Use GPU if available else revert to CPU

parser = argparse.ArgumentParser(description='Video action recogniton training')
parser.add_argument('--logfile_name', type=str, default="bi-lstm_seen51_training",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=2,
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
num_classes = 51
# if dataset == 'hmdb51':
#     num_classes=51
# elif dataset == 'ucf101':
#     num_classes = 101
# else:
#     print('We only implemented hmdb and ucf datasets.')
#     raise NotImplementedError

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
save_dir = os.path.join(save_dir_root, 'run', log_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

# pdb.set_trace()
def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    model = ConvLSTM(
        num_classes=num_classes,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    cls_criterion = nn.CrossEntropyLoss().to(device)    
    # criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
    #                                       gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    cls_criterion.to(device)

    # log_dir = os.path.join(save_dir, 'models',exp_name ,datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=100, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='test_seen',  clip_len=16), batch_size=100, num_workers=4)
    # test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=100, num_workers=4)
    test_dataloader = val_dataloader


    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    lab_list = []
    pred_list = []

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in (trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.permute(0,2,1,3,4)
                image_sequences = Variable(inputs.to(device), requires_grad=True)
                labels = Variable(labels.to(device), requires_grad=False)                

                # inputs = Variable(inputs, requires_grad=True).to(device)
                # labels = Variable(labels).to(device)
                optimizer.zero_grad()
                model.lstm.reset_hidden_state()

                if phase == 'train':
                    # pdb.set_trace()
                    predictions, lstm_out = model(image_sequences)
                else:
                    with torch.no_grad():
                        predictions, lstm_out = model(image_sequences)

                # loss = criterion(outputs, labels)
                loss = cls_criterion(predictions, labels)
                acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
                probs = nn.Softmax(dim=1)(predictions)
                preds = torch.max(probs, 1)[1]
                # pdb.set_trace()

                lab_list += labels.cpu().numpy().tolist()
                pred_list += preds.cpu().numpy().tolist()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            conf_mat = confusion_matrix(lab_list, pred_list)
            # np.save("{}.npy".format(os.path.join(save_dir, saveName + '_epoch-' + str(epoch))+'_'+ phase), conf_mat)
            # fig = plt.figure()
            # plt.imshow(conf_mat)
            # writer.add_figure('conf_mat_'+phase, fig, epoch)
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        # if useTest and epoch % test_interval == (test_interval - 1):
        #     model.eval()
        #     start_time = timeit.default_timer()

        #     running_loss = 0.0
        #     running_corrects = 0.0

        #     for inputs, labels in (test_dataloader):
        #         # print(inputs.shape)
        #         inputs = inputs.permute(0,2,1,3,4)
        #         image_sequences = Variable(inputs.to(device), requires_grad=False)
        #         labels = Variable(labels.to(device), requires_grad=False)                

        #         with torch.no_grad():
        #             model.lstm.reset_hidden_state()
        #             outputs, lstm_out = model(image_sequences)
        #             # predictions = model(image_sequences)
        #         probs = nn.Softmax(dim=1)(outputs)
        #         preds = torch.max(probs, 1)[1]
        #         loss = cls_criterion(outputs, labels)

        #         running_loss += loss.item() * inputs.size(0)
        #         running_corrects += torch.sum(preds == labels.data)

        #     epoch_loss = running_loss / test_size
        #     epoch_acc = running_corrects.double() / test_size

        #     writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
        #     writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

        #     print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        #     stop_time = timeit.default_timer()
        #     print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")
    send_dipesh("--- UCF code ENDED ---")
    
