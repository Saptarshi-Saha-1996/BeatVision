import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECGNET(nn.Module):
    def __init__(self):
        super(ECGNET, self).__init__()
        
        # Encoder
        self.PreProcess = nn.Sequential()
        self.PreProcess.add_module("Conv", nn.Conv2d(in_channels=1, out_channels=4,
                                                      kernel_size=(3,3), stride=(1,1), padding=(2,1), bias=True)) # 1,30,512 -> 4,32,512
        self.PreProcess.add_module("BN", nn.BatchNorm2d(num_features=4, momentum=0.1))
        self.PreProcess.add_module("ReLu", nn.GELU(approximate='tanh'))
        self.PreProcess.add_module("MaxPool", nn.AdaptiveMaxPool2d(output_size=(32, 256))) # 4,32,512 -> 4,32,256
        
        self.Convolutional_Layer1 = nn.Sequential()
        self.Convolutional_Layer1.add_module("BN1", nn.BatchNorm2d(num_features=4, momentum=0.1))
        self.Convolutional_Layer1.add_module("ReLu1", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer1.add_module("Conv1", nn.Conv2d(in_channels=4, out_channels=16,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)) # 4,32,256 -> 16,16,256
        self.Convolutional_Layer1.add_module("MaxPool1", nn.AdaptiveMaxPool2d(output_size=(16, 64))) # 16,16,256 -> 16,16,64

        self.Convolutional_Layer2 = nn.Sequential()
        self.Convolutional_Layer2.add_module("BN2", nn.BatchNorm2d(num_features=16, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu2", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv2", nn.Conv2d(in_channels=16, out_channels=32,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)) # 16,16,64 -> 32,16,64
        self.Convolutional_Layer2.add_module("Dropout1", nn.Dropout(p=0.3, inplace=True))
        self.Convolutional_Layer2.add_module("BN3", nn.BatchNorm2d(num_features=32, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu3", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv3", nn.Conv2d(in_channels=32, out_channels=64,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)) # 32,16,64 -> 64,16,64
        self.Convolutional_Layer2.add_module("Dropout2", nn.Dropout(p=0.3, inplace=True))
        self.Convolutional_Layer2.add_module("BN4", nn.BatchNorm2d(num_features=64, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu4", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv4", nn.Conv2d(in_channels=64, out_channels=128,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)) # 64,16,64 -> 128,16,64
        self.Convolutional_Layer2.add_module("MaxPool2", nn.AdaptiveMaxPool2d(output_size=(16, 32))) # 128,16,64 -> 128,16,32
        self.Convolutional_Layer2.add_module("Dropout3", nn.Dropout(p=0.3, inplace=True))
        self.Convolutional_Layer2.add_module("BN5", nn.BatchNorm2d(num_features=128, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu5", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv5", nn.Conv2d(in_channels=128, out_channels=256,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)) # 128,16,32 -> 256,16,32
        self.Convolutional_Layer2.add_module("MaxPool3", nn.AdaptiveMaxPool2d(output_size=(8, 16))) # 256,16,32 -> 256,8,16
        
        # Decoder
        self.decoder = Decoder()

        for m in self.modules():            # Weight initialisation 
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))   # Glorot initialisation for conv layer' weights
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)         # weight initialisation for BN layer
                m.bias.data.zero_()

    def forward(self, input_image):
        batch_size = input_image.size(0)
        pre_features = self.PreProcess(input_image)
        features_1 = self.Convolutional_Layer1(pre_features)
        features_2 = self.Convolutional_Layer2(features_1)
        features = self.decoder(features_2)
        return features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.Convolutional_Layer2 = nn.Sequential()
        self.Convolutional_Layer2.add_module("Conv5", nn.Conv2d(in_channels=256, out_channels=128,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True))
        self.Convolutional_Layer2.add_module("BN5", nn.BatchNorm2d(num_features=128, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu5", nn.GELU(approximate='tanh')) 
        self.Convolutional_Layer2.add_module("Conv4", nn.Conv2d(in_channels=128, out_channels=64,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True))
        self.Convolutional_Layer2.add_module("BN4", nn.BatchNorm2d(num_features=64, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu4", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv3", nn.Conv2d(in_channels=64, out_channels=32,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True))
        self.Convolutional_Layer2.add_module("BN3", nn.BatchNorm2d(num_features=32, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu3", nn.GELU(approximate='tanh'))
        self.Convolutional_Layer2.add_module("Conv2", nn.Conv2d(in_channels=32, out_channels=16,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True))
        self.Convolutional_Layer2.add_module("BN2", nn.BatchNorm2d(num_features=16, momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu2", nn.GELU(approximate='tanh'))

        self.Convolutional_Layer1 = nn.Sequential()
        self.Convolutional_Layer1.add_module("Conv1", nn.Conv2d(in_channels=16, out_channels=4,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True))
        self.Convolutional_Layer1.add_module("BN1", nn.BatchNorm2d(num_features=4, momentum=0.1))
        self.Convolutional_Layer1.add_module("ReLu1", nn.GELU(approximate='tanh'))
        
        self.PostProcess = nn.Sequential()
        self.PostProcess.add_module("Conv", nn.Conv2d(in_channels=4, out_channels=1,
                                                      kernel_size=(3,3), stride=(1,1), padding=(2,1), bias=True))
        self.PostProcess.add_module("BN", nn.BatchNorm2d(num_features=1, momentum=0.1))
        self.PostProcess.add_module("ReLu", nn.GELU(approximate='tanh'))
        
    def forward(self, features):
        x = features.view(features.size(0), 256, 2, 8)
        x = self.Convolutional_Layer2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # used bilinear interpolation
        x = self.Convolutional_Layer1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.PostProcess(x)
        return x


