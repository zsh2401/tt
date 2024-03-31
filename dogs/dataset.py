import torchvision
from PIL import Image
from torch.utils.data import Dataset

with open("./dataset/labels.csv", "r") as f:
    lines = f.readlines()[1:]
data = [(line.split(",")[0], line.split(",")[1].strip()) for line in lines]
classes = [v for v in set([line.split(",")[1].strip() for line in lines])]
classes.sort()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(classes)


def id2label(id):
    # return numeric_labels
    return classes[id]


def label2id(label):
    for i in range(len(classes)):
        if label == classes[i]:
            return numeric_labels[i]

    raise "Label not found"


# print(label2id("affenpinscher"))

class DogDataset(Dataset):
    def __init__(self, start_percent, end_percent) -> None:
        super().__init__()
        start_index = int(len(data) * start_percent)  # 计算 60% 的位置
        end_index = int(len(data) * end_percent)  # 计算 80% 的位置
        self.data = data[start_index:end_index]
        self.transform = torchvision.transforms.Compose([
            # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
            # 然后，缩放图像以创建224x224的新图像
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                     ratio=(3.0 / 4.0, 4.0 / 3.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            # 随机更改亮度，对比度和饱和度
            torchvision.transforms.ColorJitter(brightness=0.4,
                                               contrast=0.4,
                                               saturation=0.4),
            # 添加随机噪声
            torchvision.transforms.ToTensor(),
            # 标准化图像的每个通道
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, label = self.data[idx]
        image = Image.open(f"./dataset/train/{image_id}.jpg").convert("RGB")
        tensor = self.transform(image)
        return label2id(label), tensor
