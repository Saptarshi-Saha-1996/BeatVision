from typing import Any, Optional
import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassAccuracy, MulticlassSpecificity, MulticlassAUROC, MulticlassF1Score
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from ecgnet import ECGNET
from sklearn.metrics import confusion_matrix


# /home/saptarshi/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth

#backbone=resnet50(weights="IMAGENET1K_V2")
#print(backbone.fc.in_features)
#layers = list(backbone.children())[:-1]
#feature_extractor = nn.Sequential(*layers)
#print(feature_extractor)
#print(torch.nn.Sequential(*(list(resnet50.children())[:-1])))

class CF_Explainer(pl.LightningModule):
    def __init__(self, 
                 num_classes,
                 lr,
                 mix_up=True):
        super().__init__()
        self.num_classes = num_classes
        #resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #backbone = resnet50(weights="IMAGENET1K_V2")
        #num_filters = backbone.fc.in_features
        #layers = list(backbone.children())[:-1]
        self.feature_extractor = ECGNET() #nn.Sequential(*layers)
        self.classifier = nn.Sequential( nn.Linear(in_features=1024,out_features=128,bias=True),
                                        nn.Dropout(p=0.3),
                                        nn.LeakyReLU(inplace=True,negative_slope=0.01),
                                        nn.Linear(in_features=128,out_features=self.num_classes,bias=True),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()


        self.criterion =   ClsCriterion()                                      #nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.auroc= MulticlassAUROC(num_classes=self.num_classes)
        self.specificity= MulticlassSpecificity(num_classes=self.num_classes,average='weighted')
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.lr = lr
        self.adjust_lr = [1000]
        self.optimizer = 'Adam'
        self.mix_up = mix_up
        self.save_hyperparameters()

    def forward(self, x) :
        representation =  self.feature_extractor(x) #.flatten(1)          #nn.ReLU(inplace=True)(self.backbone(x))
        y =  torch.nn.LogSoftmax(dim=1)(self.classifier(representation))
        return (representation, y)
    
    def training_step(self,batch) :
        (x, target), _ = batch
        count= torch.bincount(target)
        weight = len(target)/(len(count)*count)
        weight_vector = weight[target]


        target_ = torch.nn.functional.one_hot(target,num_classes=self.num_classes)
        representation, pred = self.forward(x)

        if self.mix_up==True:
            lam = np.random.beta(0.1,0.1)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()
            mixed_image = lam * x + (1 - lam) * x[index, :]
            mixed_target  = lam * target_ + (1 - lam) * target_[index]
            mixed_representation, mix_pred = self.forward(mixed_image)
            mixed_weight_vector = lam * weight_vector+ (1 - lam) * weight_vector[index]
            loss = self.criterion(pred,target_,batch_weight = weight_vector) + self.criterion(mix_pred,mixed_target, batch_weight = mixed_weight_vector)
        else :
            loss = self.criterion(pred,target_, batch_weight = weight_vector )
        acc = self.accuracy(pred, target)
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss
    

    def on_validation_epoch_start(self):
        self.pred =[]
        self.target=[]
        return 
    
    def validation_step(self, batch,batch_idx) :
        (x, target), _ = batch
        target_ = torch.nn.functional.one_hot(target,num_classes=self.num_classes)
        representation, pred = self.forward(x)
        loss = self.criterion(pred,target_) 
        self.log("val/loss", loss)
        self.target.append(target)
        self.pred.append(pred)

        

    def on_validation_epoch_end(self):
        acc = self.accuracy(torch.cat(self.pred), torch.cat(self.target))
        auroc= self.auroc(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        specificity =self.specificity(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        f1_score = self.f1_score(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        self.log("val/acc", acc)
        self.log("val/auroc", auroc)
        self.log("val/specificity", specificity)
        self.log("val/f1_score",f1_score)
        return 
    

    def on_test_epoch_start(self) :
        self.pred =[]
        self.target=[]
        return 

    def test_step(self, batch,batch_idx) :
        (x, target), _ = batch
        target_ = torch.nn.functional.one_hot(target,num_classes=self.num_classes)
        representation, pred = self.forward(x)
        loss = self.criterion(pred,target_) 
        self.log("val/loss", loss)
        self.target.append(target)
        self.pred.append(pred)

    def on_test_epoch_end(self) :
        acc = self.accuracy(torch.cat(self.pred), torch.cat(self.target))
        auroc= self.auroc(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        specificity =self.specificity(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        f1_score = self.f1_score(torch.cat(self.pred).squeeze(), torch.cat(self.target).squeeze())
        
        self.log("test/acc", acc)
        self.log("test/auroc", auroc)
        self.log("test/specificity", specificity)
        self.log("test/f1_score",f1_score)


    def configure_optimizers(self) :
        if self.optimizer == 'SGD':
            opt =  torch.optim.SGD(params=[*self.feature_extractor.parameters(),*self.classifier.parameters()],lr=self.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        elif self.optimizer == 'Adam':
            opt = torch.optim.Adam(params=[*self.feature_extractor.parameters(),*self.classifier.parameters()],
                                lr=self.lr, amsgrad=True,
                                weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.adjust_lr,gamma=0.1)
        return [opt],[scheduler] 
    
    def checkpoint(self):
        return ModelCheckpoint(monitor='val/auroc',mode="max",save_top_k=1,save_last=True)
    




class ClsCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label, batch_weight=None):
        """
        :param predict: B*C log_softmax result
        :param label: B*C one-hot label
        :param batch_weight: B*1 0-1 weight for each item in a batch
        :return: cross entropy loss
        """
        if batch_weight is None:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1))
        else:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1) * batch_weight)
        return cls_loss
    



