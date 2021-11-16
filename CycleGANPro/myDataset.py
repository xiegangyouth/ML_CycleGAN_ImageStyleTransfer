# 1、导入包
import os
import glob
import random
from PIL import  Image
import torchvision.transforms as transforms
from torch.utils.data import  Dataset,DataLoader

class myDataset(Dataset):
    def __init__(self, root = "", transform = None, model = "train"):
        self.transform = transform

        self.pathA = os.path.join(root, model, "A/*")
        self.pathB = os.path.join(root, model, "B/*")

        self.listA = glob.glob(self.pathA)
        self.listB = glob.glob(self.pathB)

    def __getitem__(self, i):
        img_pathA = self.listA[i % len(self.listA)]
        img_pathB = random.choice(self.listB)

        imgA = Image.open(img_pathA)
        imgB = Image.open(img_pathB)

        return {"A":self.transform(imgA), "B":self.transform(imgB)}

    def __len__(self):
        return max(len(self.listA), len(self.listB))

if __name__ == "__main__":
    root = "datasets/apple2orange"
    mytransforms = transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor()
    ])
    imageDataset = myDataset(root, mytransforms, "train")
    imageDataloader = DataLoader(imageDataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(imageDataloader):
        print(i)
        print(data)
