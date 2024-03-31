<<<<<<< HEAD
#!/usr/bin/env python
import os

import torch
=======
import os

import torch
import torchvision
>>>>>>> 0ecfa60fad91720c2a5ec56a65cd6a74112cff1f

import torchvision.transforms as VT
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

<<<<<<< HEAD
from dataset import id2label
=======
from cifar10.model import Cifar18
from cifar10.dataset import id2label
>>>>>>> 0ecfa60fad91720c2a5ec56a65cd6a74112cff1f

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

<<<<<<< HEAD
batch_size =90
# 默认超参最好的版本
pth_name = "checkpoints/2024-04-01-00_26_38-4-64.35-0.0051.pth"
checkpoint = torch.load(pth_name)
=======
batch_size = 1
pth_name = "2024-03-31-22_27_56-0-67.11-0.0309.pth"
checkpoint = torch.load(pth_name, map_location=device)
>>>>>>> 0ecfa60fad91720c2a5ec56a65cd6a74112cff1f

model = checkpoint["model"]
model.to(device)
model.eval()
print(f"Loaded {pth_name}")


# class Testset(torch.utils.DataSet)

class Testset(Dataset):
    def __init__(self):
        self.images = [(name[:-4], f"./dataset/test/{name}") for name in os.listdir("./dataset/test")]
<<<<<<< HEAD
        self.transform = VT.Compose([
            VT.Resize(256),
    # 从图像中心裁切224x224大小的图片
            VT.CenterCrop(224),
            VT.ToTensor(),
            VT.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])
=======
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            # 从图像中心裁切224x224大小的图片
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
>>>>>>> 0ecfa60fad91720c2a5ec56a65cd6a74112cff1f

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        id, path = self.images[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return id, image


test_loader = DataLoader(Testset(), shuffle=True, batch_size=batch_size)

with open("submission.csv", "w") as f:
    f.write("id,label")
    for ids, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
<<<<<<< HEAD
        preds = []
        with torch.no_grad():
            outputs = model(images)
        preds.extend(outputs.argmax(dim=1).type(torch.int32).cpu().numpy())
        for i, prediction in enumerate(preds):
            label = id2label(prediction)
=======
        with torch.no_grad():
            outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        for i, prediction in enumerate(predictions):
            label = id2label(prediction.int())
>>>>>>> 0ecfa60fad91720c2a5ec56a65cd6a74112cff1f
            f.write(f"\n{ids[i]},{label}")
