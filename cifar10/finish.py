import os

import torch

import torchvision.transforms as VT
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from cifar10.model import Cifar18
from cifar10.dataset import id2label

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

batch_size = 1
pth_name = "2024-03-31-22_27_56-0-67.11-0.0309.pth"
checkpoint = torch.load(pth_name, map_location=device)

model = checkpoint["model"]
model.to(device)
model.eval()
# model.no_
print(f"Loaded {pth_name}")


# class Testset(torch.utils.DataSet)

class Testset(Dataset):
    def __init__(self):
        self.images = [(name[:-4], f"./dataset/test/{name}") for name in os.listdir("./dataset/test")]
        self.transform = VT.Compose([
            VT.Resize(40),
            VT.ToTensor(),
            VT.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

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
        with torch.no_grad():
            outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        for i, prediction in enumerate(predictions):
            label = id2label(prediction.int())
            f.write(f"\n{ids[i]},{label}")
