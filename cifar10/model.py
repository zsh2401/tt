import torch.nn
import torchvision


class Cifar18(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
