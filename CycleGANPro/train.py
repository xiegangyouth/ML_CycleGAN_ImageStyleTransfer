import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models_densenet import Discriminator, Generator # 基于DenseNet
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from myDataset import myDataset
import itertools
import tensorboardX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 1
size = 256
lr = 0.0002
n_epoch = 100
epoch = 0
decay_epoch = 100

# 创建生成器和判别器
netG_A2B = Generator(k=4, depth=50, reduction=0.4).to(device)
netG_B2A = Generator(k=4, depth=50, reduction=0.4).to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

# 定义损失函数
loss_GAN = torch.nn.MSELoss() #对抗损失损失:均方误差
loss_cycle = torch.nn.L1Loss() #循环一致性损失:1范式
loss_identity = torch.nn.L1Loss() #样本相似损失:1范式

#优化器和衰减规则
opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.9999))
opt_DA = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.9999))
opt_DB = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.9999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)
lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(n_epoch,  epoch,  decay_epoch).step)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(n_epoch, epoch, decay_epoch).step)

image_root =  "datasets/apple2orange"
input_A = torch.ones([batchsize, 3, size, size],
                     dtype=torch.float).to(device)
input_B = torch.ones([batchsize, 3, size, size],
                     dtype=torch.float).to(device)
label_real = torch.ones([1], dtype=torch.float,
                        requires_grad=False).to(device)
label_fake = torch.zeros([1], dtype=torch.float,
                        requires_grad=False).to(device)

buffer_f_A = ReplayBuffer()
buffer_f_B = ReplayBuffer()

# 日志目录
log_path = "logs"
writer_log = tensorboardX.SummaryWriter(log_path)

# 图像增强操作
transforms_ = [
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(myDataset(image_root, transforms_, "train"),
                        batch_size=batchsize, shuffle=True, num_workers=8)
# 主方法
def main():
    step = 0
    for epoch in range(n_epoch):
        for i, batch in enumerate(dataloader):
            # 获取真实的A和B
            real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
            real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

            # ------------------------------------生成器梯度清零-----------------------------------------
            opt_G.zero_grad()

            same_B = netG_A2B(real_B)
            loss_identity_B = loss_identity(same_B, real_B) * 5.0

            #
            same_A = netG_B2A(real_A)
            loss_identity_A = loss_identity(same_A, real_A) * 5.0

            # A2B对抗损失
            fake_B = netG_A2B(real_A) #生成器：利用真实A生成虚假B
            pred_fake = netD_B(fake_B) #判别器：判断虚假B，输出其为真实B的概率值
            loss_GAN_A2B = loss_GAN(pred_fake, label_real) #计算A2B对抗损失

            # B2A对抗损失
            fake_A = netG_B2A(real_B)#生成器：利用真实B生成虚假A
            pred_fake = netD_A(fake_A)#判别器：判断虚假A，输出其为真实A的概率值
            loss_GAN_B2A = loss_GAN(pred_fake, label_real)#计算B2A对抗损失

            # 循环一致性损失
            recovered_A = netG_B2A(fake_B)#生成器：将虚假B恢复成A
            loss_cycle_ABA = loss_cycle(recovered_A, real_A) * 10.0 #计算前向循环一致性损失
            recovered_B = netG_A2B(fake_A)#生成器：将虚假A恢复成B
            loss_cycle_BAB = loss_cycle(recovered_B, real_B) * 10.0#计算后向循环一致性损失

            #对抗总损失
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # 反向传播，更新参数
            loss_G.backward()
            opt_G.step()

            #-------------------------------------判别器梯度清零-----------------------------------------
            opt_DA.zero_grad()

            #
            pred_real = netD_A(real_A)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_A = buffer_f_A.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)

            # 判别器1总损失
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            # 反向传播，更新参数
            loss_D_A.backward()
            opt_DA.step()

            ####B--->
            opt_DB.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = loss_GAN(pred_real, label_real)

            fake_B = buffer_f_B.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = loss_GAN(pred_real, label_fake)

            # 判别器2总损失
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_DB.step()

            # 控制台输出打印
            print("------------{}------------:".format(epoch+1))
            print("loss_G:{}, loss_G_identity:{}, loss_G_GAN:{}, loss_G_cycle:{}, loss_D_A:{}, loss_D_B:{}".format(
                loss_G, loss_identity_A + loss_identity_A, loss_GAN_A2B + loss_GAN_B2A, loss_cycle_BAB + loss_cycle_ABA, loss_D_A, loss_D_B
            ))

            # 写入日志
            writer_log.add_scalar("loss_G", loss_G, global_step=step + 1)
            writer_log.add_scalar("loss_G_identity", loss_identity_A + loss_identity_A, global_step=step + 1)
            writer_log.add_scalar("loss_G_GAN", loss_GAN_A2B + loss_GAN_B2A, global_step=step + 1)
            writer_log.add_scalar("loss_G_cycle", loss_cycle_BAB + loss_cycle_ABA, global_step=step + 1)
            writer_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
            writer_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)

            step += 1

        # 更新衰减系数
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()
        lr_scheduler_G.step()

        # 保存预训练模型
        torch.save(netG_A2B.state_dict(), 'models_DENS/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'models_DENS/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'models_DENS/netD_A.pth')
        torch.save(netD_B.state_dict(), 'models_DENS/netD_B.pth')

if __name__ == "__main__":
    main()










