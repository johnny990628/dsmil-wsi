import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from collections import OrderedDict

class ResNetSimCLR(nn.Module):

    def __init__(self, fine_tune_from, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        pretrained = fine_tune_from=='imagenet'
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained)} 
        if not pretrained:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained,norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=pretrained,norm_layer=nn.InstanceNorm2d)} 

        resnet = self._get_basemodel(base_model)
        if pretrained:
            resnet = self._replace_norm_layer(resnet, nn.InstanceNorm2d)

        num_ftrs = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _replace_norm_layer(self, model, norm_layer):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, norm_layer(module.num_features, affine=True))
            elif len(list(module.children())) > 0:
                self._replace_norm_layer(module, norm_layer)
        return model


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
