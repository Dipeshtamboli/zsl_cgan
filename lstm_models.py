import os
from old_16_frames import VideoDataset
from torch.utils.data import DataLoader
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet152

##############################
#         Encoder
##############################

# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Encoder, self).__init__()
#         resnet = resnet152(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#         self.final = nn.Sequential(
#             nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
#         )

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         with torch.no_grad():
#             x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)
#         return self.final(x).reshape(batch_size, seq_length, c, h, w)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)


##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


##############################
#         ConvLSTM
##############################


class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024,out_feat_dim=256, bidirectional=True, attention=True
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            # nn.Linear(hidden_dim, num_classes),
            nn.Linear(hidden_dim, out_feat_dim),
            # nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        # print(x.shape)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            # print(attention_w.shape)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
            # print(x.shape)
        else:
            x = x[:, -1]
        return self.output_layers(x), x
##############################
#  Semantic Fully Connected
##############################
class Semantic_FC(nn.Module):
    def __init__(self):
        super(Semantic_FC, self).__init__()
        # resnet = resnet152(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final_out = nn.Sequential(
            nn.Linear(300, 256), 
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Linear(256, 128), 
        )

    def forward(self, x):
        x = self.final_out(x)
        x = x.view(x.size(0), -1)
        return (x)

##############################
#  Final classifier
##############################
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # resnet = resnet152(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier_out = nn.Sequential(
            nn.Linear(256+256, 256), 
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Linear(256, num_classes), 
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.classifier_out(x)
        x = x.view(x.size(0), -1)
        return (x)
##############################
#     Conv2D Classifier
#        (Baseline)
##############################


class ConvClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(ConvClassifier, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        # print("batch:", batch_size, "seq_length:", seq_length)
        # print("feature", x.shape)
        x = x.view(batch_size * seq_length, -1)
        # print("view:", x.shape)
        x = self.final(x)
        # print("final:", x.shape)
        x = x.view(batch_size, seq_length, -1)
        # print("final_view:", x.shape)
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    train_dataloader = DataLoader(VideoDataset(dataset='ucf101', split='train',clip_len=16), batch_size=20, shuffle=True, num_workers=4)
    model = ConvLSTM(
        num_classes=101,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        out_feat_dim=256,
        bidirectional=True,
        attention=True,
    )
    model.to('cuda')
    for inputs, labels in (train_dataloader):
        inputs = inputs.permute(0,2,1,3,4)
        image_sequences = Variable(inputs.to("cuda"), requires_grad=True)        
        out = model(image_sequences)
        pdb.set_trace()