import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy
from nets import *
from tensorboardX import SummaryWriter

import torchvision.models as models
import torchvision.transforms as transforms

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
        labels = Variable(labels.data, requires_grad=False).cuda()
        outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
        labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
        return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
        if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
        def __init__(self, classes, classes_map, args):
		# Hyper Parameters
                self.args = args
                self.save_epoch = args.snapshot
                self.init_lr = 1e-3
                self.num_epochs = args.num_epochs
                #self.num_epochs = 1
                self.batch_size = args.batch_size
                self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)] #hardcoded decay schedule
                self.lr_dec_factor = 10
		
                self.pretrained = False
                self.momentum = 0.9
                self.weight_decay = 0.0001
		# Constant to provide numerical stability while normalizing
                self.epsilon = 1e-16
                self.semantic_dim = 300

                current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                save_dir_root = current_dir
                log_name = args.logfile_name
                self.save_dir = os.path.join(save_dir_root, 'run', log_name)
                log_dir = os.path.join(self.save_dir)
                #self.writer = SummaryWriter(log_dir=log_dir)

                self.att = np.load("../npy_files/seen_semantic_51.npy")
                self.att = torch.tensor(self.att).cuda()    

		# Network architecture
                super(Model, self).__init__()
                
                self.conv_lstm = ConvLSTM(
                num_classes=classes,
                latent_dim=512,
                lstm_layers=1,
                hidden_dim=1024,
                bidirectional=True,
                attention=True,
                )

                self.semantic_dim = 300
                self.model = Classifier(num_classes = classes, semantic_dim=self.semantic_dim)

		#self.model = models.resnet34(pretrained=self.pretrained)
		#self.model.apply(kaiming_normal_init)
                states = self.model.state_dict()
                w_name = [*states][-2]
                b_name = [*states][-1]
                last_weights = states[w_name]
                last_biases = states[b_name]   
                self.num_features = last_weights.size(1)
		
                #num_features = self.model.fc.in_features
                self.model.classifier_out = nn.Linear(self.num_features, classes, bias=False)
                self.fc = self.model.classifier_out
                self.feature_extractor = self.model.extractor
                #self.feature_extractor = nn.DataParallel(self.feature_extractor) 


		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
                self.n_classes = 0
                self.n_known = 0
                self.classes_map = classes_map

        def forward(self, x, y):
                x = torch.cat((x.view(x.size(0), -1), y.view(y.size(0), -1)), -1)
                x = self.feature_extractor(x)
                x = self.fc(x)
                x = x.view(x.size(0), -1)
                return x

        def increment_classes(self, new_classes):
                """Add n classes in the final fc layer"""
                n = len(new_classes)
                print('new classes: ', n)
                in_features = self.fc.in_features
                out_features = self.fc.out_features
                weight = self.fc.weight.data

                if self.n_known == 0:
                        new_out_features = n
                else:
                        new_out_features = out_features + n
                print('new out features: ', new_out_features)
                self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
                self.fc = self.model.fc
		
                kaiming_normal_init(self.fc.weight)
                self.fc.weight.data[:out_features] = weight
                self.n_classes += n

        def classify(self, images, labels):
                """Classify images by softmax

                Args:
                        x: input image batch
                Returns:
                        preds: Tensor of size (batch_size,)
                """
                true_features = self.conv_lstm(images)
                semantic = self.att[labels]
                _, preds = torch.max(torch.softmax(self.forward(true_features.float(), semantic.float()), dim=1), dim=1, keepdim=False)

                return preds

        def update(self, dataset, class_map, args):

                self.compute_means = True

		# Save a copy to compute distillation outputs
                prev_conv_lstm = copy.deepcopy(self.conv_lstm)
                prev_conv_lstm.cuda()
                prev_feature_extractor = copy.deepcopy(self.feature_extractor)
                prev_feature_extractor.cuda()
                prev_fc = copy.deepcopy(self.fc)
                prev_fc.cuda()

                classes = list(set(dataset.train_labels))
		#print("Classes: ", classes)
                print('Known: ', self.n_known)
                if self.n_classes == 10 and self.n_known == 0:
                        new_classes = [classes[i] for i in range(1,len(classes))]
                else:
                        new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

                if len(new_classes) > 0:
                        self.increment_classes(new_classes)
                        self.cuda()

                loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

                print("Batch Size (for n_classes classes) : ", len(dataset))
                optimizer = optim.SGD(list(self.conv_lstm.parameters()) + list(self.model.parameters()), lr=self.init_lr, momentum = self.momentum, weight_decay=self.weight_decay)

                with tqdm(total=self.num_epochs) as pbar:
                        for epoch in range(self.num_epochs):
								
                                for i, (indices, images, labels) in enumerate(loader):
                                        optimizer.zero_grad()
                                        images = images.permute(0,2,1,3,4)
                                        seen_labels = []
                                        images = Variable(torch.FloatTensor(images)).cuda()
                                        seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                                        labels = Variable(seen_labels).cuda()
                                        true_features = self.conv_lstm(images)
                                        semantic = self.att[labels]
					# indices = indices.cuda()

                                        logits = self.forward(true_features.float(), semantic.float())
                                        cls_loss = nn.CrossEntropyLoss()(logits, labels)
                                        if self.n_classes//len(new_classes) > 1:
                                                dist_target = prev_conv_lstm(images)
                                                dist_target, semantic = dist_target.float(), semantic.float()
                                                dist_target = torch.cat((dist_target.view(dist_target.size(0), -1), semantic.view(semantic.size(0), -1)), -1)
                                                dist_target = prev_feature_extractor(dist_target)
                                                dist_target = dist_target.view(dist_target.size(0), -1)
                                                dist_target = prev_fc(dist_target)
                                                logits_dist = logits[:,:-(self.n_classes-self.n_known)]
                                                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                                                loss = dist_loss+cls_loss
                                        else:
                                                loss = cls_loss

                                        loss.backward()
                                        optimizer.step()

                                        #self.writer.add_scalar('data/loss_epoch', loss, epoch)

                                        if (i+1) % 1 == 0:
                                                tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data))

                                save_path = os.path.join(self.save_dir, 'UCF101-BiLSTM' + '_epoch-' + str(epoch) + '.pth.tar')
                     
                                if epoch % self.save_epoch == (self.save_epoch - 1):
                                    torch.save({'epoch': epoch + 1, 'conv_lstm_state_dict': self.conv_lstm.state_dict(), 'extractor_state_dict': self.feature_extractor.state_dict(), 'fc_state_dict': self.fc.state_dict(),}, save_path)
                                    print("Save model at {}\n".format(save_path))


                                pbar.update(1)
                        #self.writer.close()