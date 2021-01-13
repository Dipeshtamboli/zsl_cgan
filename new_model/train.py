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
from nets import *

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
parser.add_argument('--logfile_name', type=str, default="generator_w_with_sem:10",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
args = parser.parse_args()

gpu_id = str(args.gpu)
log_name = args.logfile_name
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

# latent_dim = 300+100
b1=0.5
b2=0.999
batch_size = 100
semantic_dim = 300
noise_dim = 100
input_dim = 2048
nEpochs = 200  # Number of epochs for training
# nEpochs = 10  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 10 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101
num_classes = 10

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
save_dir = os.path.join(save_dir_root, 'run', log_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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
    classifier = Classifier(num_classes = num_classes, semantic_dim=semantic_dim)
    generator = Generator(semantic_dim, noise_dim)
    discriminator = Discriminator(input_dim=input_dim, semantic_dim=semantic_dim)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=lr)

    if cuda:
        model = model.to(device)
        classifier = classifier.to(device)
        generator.cuda()
        discriminator.cuda()

    cls_criterion = nn.CrossEntropyLoss().to(device)
    adversarial_loss = torch.nn.MSELoss().to(device)

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

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='data_1_train',clip_len=16), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='data_1_test',  clip_len=16), batch_size=batch_size, num_workers=4)
    # test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=100, num_workers=4)
    test_dataloader = val_dataloader


    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    lab_list = []
    pred_list = []

    att = np.load("../npy_files/seen_semantic_51.npy")
    att = torch.tensor(att).cuda()    

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        # for phase in ['train', 'val']:
        start_time = timeit.default_timer()

            # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            # if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # scheduler.step()
        model.train()
        classifier.train()
        generator.train()
        discriminator.train()
            # else:
            #     model.eval()

        for inputs, labels in (trainval_loaders["train"]):
            inputs = inputs.permute(0,2,1,3,4)
            image_sequences = Variable(inputs.to(device), requires_grad=True)
            labels = Variable(labels.to(device), requires_grad=False)                

            optimizer.zero_grad()
            model.lstm.reset_hidden_state()
            loop_batch_size = len(inputs)

            valid = Variable(FloatTensor(loop_batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(loop_batch_size, 1).fill_(0.0), requires_grad=False)

            optimizer_G.zero_grad()
            noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, num_classes, loop_batch_size)))
            semantic = att[gen_labels]
            semantic_true = att[labels]

            true_features_2048 = model(image_sequences)
            real_imgs = Variable(true_features_2048.type(FloatTensor))
            # pdb.set_trace()
            predictions = classifier(true_features_2048, semantic_true.type(FloatTensor))
            gen_imgs = generator(semantic.float(), noise)
            generated_preds = classifier(gen_imgs, semantic.type(FloatTensor))


            validity = discriminator(gen_imgs, semantic.type(FloatTensor))
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward(retain_graph=True)
            # g_loss.backward()
            # for name, param in generator.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.sum())
            #     else:
            #         print(name, param.grad)

            optimizer_G.step()                

            optimizer_D.zero_grad()

            validity_real = discriminator(true_features_2048, semantic_true.type(FloatTensor))
            d_real_loss = adversarial_loss(validity_real, valid)
            validity_fake = discriminator(gen_imgs.detach(), semantic.type(FloatTensor))
            d_fake_loss = adversarial_loss(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward(retain_graph=True)
            # d_loss.backward()
            
            optimizer_D.step()

            cls_loss = cls_criterion(predictions, labels)
            gen_multiclass_CEL = cls_criterion(generated_preds, gen_labels)
            # loss =  cls_loss  + 10*gen_multiclass_CEL
            loss =  cls_loss  + gen_multiclass_CEL
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            probs = nn.Softmax(dim=1)(predictions)
            preds = torch.max(probs, 1)[1]
            lab_list += labels.cpu().numpy().tolist()
            pred_list += preds.cpu().numpy().tolist()
            
            loss.backward()
            # for name, param in generator.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.sum())
            #     else:
            #         print(name, param.grad)

            # for name, param in (list(classifier.named_parameters())+list(model.named_parameters())):
            #     if param.grad is not None:
            #         print(name, param.grad.sum())
            #     else:
            #         print(name, param.grad)
            
            optimizer.step()
            # for name, param in generator.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.sum())
            #     else:
            #         print(name, param.grad)

            # exit()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        conf_mat = confusion_matrix(lab_list, pred_list)
        epoch_loss = running_loss / trainval_sizes["train"]
        epoch_acc = running_corrects.double() / trainval_sizes["train"]

        writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("train", epoch+1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        # Validation loop

        if useTest and epoch % test_interval == (test_interval - 1):
        # if True:
            model.eval()
            classifier.eval()
            generator.eval()
            discriminator.eval()            
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            gen_run_corrects = 0.0

            for inputs, labels in (test_dataloader):
                # print(inputs.shape)
                inputs = inputs.permute(0,2,1,3,4)
                image_sequences = Variable(inputs.to(device), requires_grad=False)
                labels = Variable(labels.to(device), requires_grad=False)                
                loop_batch_size = len(inputs)
                noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, num_classes, loop_batch_size)))
                semantic = att[gen_labels]
                semantic_true = att[labels]

                with torch.no_grad():
                    model.lstm.reset_hidden_state()
                    # outputs, lstm_out = model(image_sequences)
                    true_features_2048 = model(image_sequences)
                    probs = classifier(true_features_2048, semantic_true.type(FloatTensor))

                    gen_imgs = generator(semantic.float(), noise)
                    generated_probs = classifier(gen_imgs, semantic.type(FloatTensor))
                    # predictions = model(image_sequences)
                
                preds = torch.max(probs, 1)[1]

                generated_preds = torch.max(generated_probs, 1)[1]

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                gen_run_corrects += torch.sum(generated_preds == gen_labels.data)


            real_epoch_acc = running_corrects.double() / test_size
            gen_epoch_acc = gen_run_corrects.double() / test_size

            # writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            writer.add_scalar('data/gen_test_acc_epoch', gen_epoch_acc, epoch)

            print("[test] Epoch: {}/{} Gen_Acc: {} Acc: {}".format(epoch+1, nEpochs, gen_epoch_acc, real_epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        save_path = os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')
        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'extractor_state_dict': model.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, save_path)
            print("Save model at {}\n".format(save_path))

        # exit()
    writer.close()


if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")
    send_dipesh("--- UCF code ENDED ---")
    
