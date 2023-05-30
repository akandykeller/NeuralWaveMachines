from torch import nn
import torch.nn.functional as F

def LinearLayer_MLP_Encoder(in_features, out_features):
    model = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, out_features*3),
                nn.ReLU(True),
                nn.Linear(out_features*3, out_features*2),
                nn.ReLU(True),
                nn.Linear(out_features*2, out_features*2))
    return model

def LinearLayer_MLP_Decoder(in_features, out_features):
    model = nn.Sequential(
                nn.Linear(in_features, in_features*2),
                nn.ReLU(True),
                nn.Linear(in_features*2, in_features*3),
                nn.ReLU(True),
                nn.Linear(in_features*3, out_features),
                nn.Sigmoid())
    return model

def LinearFC_Encoder(in_features, out_features):
    model = nn.Sequential(
                  nn.Flatten(start_dim=1),
                  nn.Linear(in_features, out_features*2))
    return model

def LinearFC_Decoder(in_features, out_features):
    model = nn.Sequential(
                  nn.Linear(in_features, out_features),
                nn.Sigmoid())
    return model

def Identity_Encoder():
    return lambda x: x

def Identity_Decoder():
    return lambda x: x