import torch.nn as nn
import torch
import math

class ECGNET(nn.Module):
    def __init__(self, ) :
        super().__init__()
        self.PreProcess = nn.Sequential()
        self.PreProcess.add_module("Conv",nn.Conv2d(in_channels=1,out_channels=2,
                                                    kernel_size=(3,3), stride=(2,2), bias=True,padding=(1,1))) # 8,32,512
        self.PreProcess.add_module('BN',nn.BatchNorm2d(num_features=2,momentum=0.1))
        self.PreProcess.add_module('ReLu',nn.GELU(approximate='tanh')) # LeakyReLU(inplace=True,negative_slope=0.01))  # 8, 32,512
        self.PreProcess.add_module("MaxPool",nn.AdaptiveMaxPool2d(output_size=(32,512)))
        #nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1,ceil_mode=False))

        self.Convolutional_Layer1 = nn.Sequential()
        self.Convolutional_Layer1.add_module("BN1",nn.BatchNorm2d(num_features=2,momentum=0.1))
        self.Convolutional_Layer1.add_module("ReLu1",nn.GELU(approximate='tanh')) #LeakyReLU(inplace=True,negative_slope=0.01))
        self.Convolutional_Layer1.add_module("Conv1", nn.Conv2d(in_channels=2, out_channels=4,
                                                               kernel_size=(3,3), stride=2, padding=(1,1),bias=True)) # 16,16,256
        self.Convolutional_Layer1.add_module("MaxPool1", nn.AdaptiveMaxPool2d(output_size=(16,128)))  # 16, 8, 64

        self.Convolutional_Layer2 = nn.Sequential()
        self.Convolutional_Layer2.add_module("BN2",nn.BatchNorm2d(num_features=4,momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu2",nn.GELU(approximate='tanh')) #
        self.Convolutional_Layer2.add_module("Conv2", nn.Conv2d(in_channels=4, out_channels=8,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=True)) # 32, 8,128
        self.Convolutional_Layer2.add_module('Dropout1',nn.Dropout(p=0.3,inplace=True))
        self.Convolutional_Layer2.add_module("BN3",nn.BatchNorm2d(num_features=8,momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu3",nn.GELU(approximate='tanh')) # LeakyReLU(inplace=True,negative_slope=0.01))
        self.Convolutional_Layer2.add_module("Conv3", nn.Conv2d(in_channels=8, out_channels=16, 
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=True)) #64,8,128
        self.Convolutional_Layer2.add_module('Dropout2',nn.Dropout(p=0.3,inplace=True))
        self.Convolutional_Layer2.add_module("BN4",nn.BatchNorm2d(num_features=16,momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu4",nn.GELU(approximate='tanh')) #LeakyReLU(inplace=True,negative_slope=0.01))
        self.Convolutional_Layer2.add_module("Conv4", nn.Conv2d(in_channels=16, out_channels=32,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=True))# 128,8,128
        #self.Convolutional_Layer2.add_module("MaxPool2", nn.AdaptiveMaxPool2d(output_size=(8,32))) # 128,4,32
        self.Convolutional_Layer2.add_module('Dropout3',nn.Dropout(p=0.3,inplace=True))
        self.Convolutional_Layer2.add_module("BN5",nn.BatchNorm2d(num_features=32,momentum=0.1))
        self.Convolutional_Layer2.add_module("ReLu5",nn.GELU(approximate='tanh')) 
        self.Convolutional_Layer2.add_module("Conv5", nn.Conv2d(in_channels=32, out_channels=64,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=True))# 128,8,128
        
        self.Convolutional_Layer2.add_module("MaxPool2", nn.AdaptiveMaxPool2d(output_size=(4,16)))

        
        self.final_layer = nn.Sequential()
        self.final_layer.add_module('BN', nn.BatchNorm2d(num_features=64,momentum=0.1)) #nn.LayerNorm([64,4,16])) #BatchNorm2d(num_features=256,momentum=0.1))
        self.final_layer.add_module("ReLu",nn.GELU(approximate='tanh')) #LeakyReLU(inplace=True,negative_slope=0.01))
        self.final_layer.add_module("Conv",nn.Conv2d(in_channels=64, out_channels=128,
                                                               kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=True)) # 256,4,32
        self.final_layer.add_module("AvgPool", nn.AdaptiveAvgPool2d((2,4)))  # 256,2,2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.01)
                m.bias.data.zero_()

    def forward(self,input_image):
        batch_size = input_image.size(0)
        pre_features =  self.PreProcess(input_image)                        # (1,30,768)    ---> (4,32,512)
        features_1 =    self.Convolutional_Layer1(pre_features)             # (4,32,512)    --->  (16,16,128)                         
        features_2 =    self.Convolutional_Layer2(features_1)               # (16,16,128)    --->  (256,8,16)
        features = self.final_layer(features_2)                             # (256,8,16)    --->  (512,2,4)
        features = features.view(batch_size,-1)                             # (512,2,4)     ---> 2048
        return features


# class Pre_process_Net(nn.Module):
#     def __init__(self, ) :
#         super().__init__()
#         self.Net = nn.Sequential()
#         self.Net.add_module("Conv",nn.Conv2d(in_channels=3,out_channels=3,
#                                                     kernel_size=(1,7), stride=(1,7), padding=(1,1), bias=True))
#         self.Net.add_module('BN',nn.BatchNorm2d(num_features=3,momentum=0.01))
#         self.Net.add_module('ReLu',nn.LeakyReLU(inplace=True,negative_slope=0.01))                                            
#         self.Net.add_module("AvgPool", nn.AdaptiveAvgPool2d((224,224)))



#     def forward(self,input_image):
#         return self.Net(input_image)




if __name__ == "__main__":
    model = ECGNET()
    print(model)
    batch_size = 68
    imgs = torch.randn(batch_size,1,30,768)
    features = model(imgs)
    print(features.shape)