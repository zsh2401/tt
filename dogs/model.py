import torchvision
from torch import nn
from torch.nn import Module


class DogClassifier(Module):
    def __init__(self):
        super(DogClassifier, self).__init__()
        self.finetune_net = nn.Sequential()
        self.finetune_net.features = torchvision.models.resnet34(pretrained=True)
        # 定义一个新的输出网络，共有120个输出类别
        self.finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                                     nn.ReLU(),
                                                     nn.Linear(256, 120))
        # 冻结参数
        for param in self.finetune_net.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.finetune_net(x)
