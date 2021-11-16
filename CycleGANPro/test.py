import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from models_resnet import  Generator
from myDataset import myDataset
import os
from torchvision.utils import save_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 获取生成器
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
# 读取预训练模型
netG_A2B.load_state_dict(torch.load("models_RES/netG_A2B.pth"))
netG_B2A.load_state_dict(torch.load("models_RES/netG_B2A.pth"))

# 测试阶段，不需要反馈
netG_A2B.eval()
netG_B2A.eval()

size = 256
input_A = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)
input_B = torch.ones([1, 3, size, size],
                     dtype=torch.float).to(device)


transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

data_root = "datasets/apple2orange"
dataloader = DataLoader(myDataset(data_root,
                                     transforms_ , "test"),
                        batch_size=1, shuffle=False,
                        num_workers=8)

def fun():

    # 创建输出文件夹
    if not os.path.exists("outputs/A"):
        os.makedirs("outputs/A")
    if not os.path.exists("outputs/B"):
        os.makedirs("outputs/B")

    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        save_image(fake_A, "outputs/A/{}.png".format(i))
        save_image(fake_B, "outputs/B/{}.png".format(i))
        print(i)

if __name__ == "__main__":
    fun()