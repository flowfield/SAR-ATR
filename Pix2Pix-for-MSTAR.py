import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

class FacadesDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super(FacadesDataset, self).__init__()

        self.transform = transform
        x_path, l_path = [], []

        if train is True:
            xbase_path = './datasets/Facades/trainA/'
            lbase_path = './datasets/Facades/trainB/'
        else:
            xbase_path = './datasets/Facades/testA/'
            lbase_path = './datasets/Facades/testB/'

        folderpaths_x = [os.path.join(xbase_path, file_name) for file_name in os.listdir(xbase_path)]
        folderpaths_l = [os.path.join(lbase_path, file_name) for file_name in os.listdir(lbase_path)]

        x_path.extend(folderpaths_x)
        l_path.extend(folderpaths_l)

        self.X = x_path
        self.L = l_path

    def __getitem__(self, i):
        train = self.X[i]
        label = self.L[i]

        x = Image.open(train)
        l = Image.open(label)

        if self.transform is not None:
            x = self.transform(x)
            l = self.transform(l)

        return x, l

    def __len__(self):
        return len(self.X)

def dataloader_train(batch_size):
    transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    train_dataset = FacadesDataset(train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader

def dataloader_test():
    transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    test_dataset = FacadesDataset(train=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return test_dataloader

class UNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, isnormalize=True, dropout=0.2):
        super(UNetDownSample, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if isnormalize:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=0.2))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        # * 가변적인 개수를 가진 위치 인수로 정의
        # ** 가변적인 개수를 가진 인수로 정의, 입력이 dictionary로 들어오므로 kwargs.items()와 같이 사용

    def forward(self, x):
        return self.model(x)

class UNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=0.2):
        super(UNetUpSample, self).__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if drop_out > 0:
            layers.append(nn.Dropout(drop_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_channel):
        x = self.model(x)
        x = torch.cat((x, skip_channel), 1)

        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDownSample(in_channels, 64, isnormalize=False, dropout=0.0)
        self.down2 = UNetDownSample(64, 128, isnormalize=True, dropout=0.0)
        self.down3 = UNetDownSample(128, 256, isnormalize=True, dropout=0.0)
        self.down4 = UNetDownSample(256, 512, isnormalize=True, dropout=0.5)
        self.down5 = UNetDownSample(512, 512, isnormalize=True, dropout=0.5)
        self.down6 = UNetDownSample(512, 512, isnormalize=True, dropout=0.5)
        self.down7 = UNetDownSample(512, 512, isnormalize=True, dropout=0.5)
        self.down8 = UNetDownSample(512, 512, isnormalize=False, dropout=0.5)

        self.up1 = UNetUpSample(512, 512, drop_out=0.5)
        self.up2 = UNetUpSample(1024, 512, drop_out=0.5)
        self.up3 = UNetUpSample(1024, 512, drop_out=0.5)
        self.up4 = UNetUpSample(1024, 512, drop_out=0.5)
        self.up5 = UNetUpSample(1024, 256, drop_out=0.0)
        self.up6 = UNetUpSample(512, 128, drop_out=0.0)
        self.up7 = UNetUpSample(256, 64, drop_out=0.0)

        self.out = nn.Sequential(nn.Upsample(scale_factor=2),
                                 nn.ZeroPad2d((1, 0, 1, 0)),
                                 nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
                                 nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.out(u7)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, isnormalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if isnormalize:
                layers.append(nn.InstanceNorm2d(out_channels))

            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*discriminator_block(in_channels*2, 64, isnormalize=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
                                   )

    def forward(self, real_img, condition_img):
        img_input = torch.cat((real_img, condition_img), 1)
        return self.model(img_input)

def weights_init_normal(model):
    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)


if __name__ == '__main__':
    in_channels = 3
    out_channels = 3
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    epochs = 1
    batch_size = 16
    sampling_interval = 100
    lambda_criterion2 = 0.5

    train_dataloader = dataloader_train(batch_size)
    val_dataloader = dataloader_test()

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    criterion1 = torch.nn.MSELoss().cuda()
    criterion2 = torch.nn.L1Loss().cuda()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    start_time = time.time()

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            real_img, label_img = data
            real_img = real_img.cuda()
            label_img = label_img.cuda()

            real = torch.cuda.FloatTensor(real_img.size(0), 1, 16, 16).fill_(1.0)
            fake = torch.cuda.FloatTensor(real_img.size(0), 1, 16, 16).fill_(0.0)

            optimizer_g.zero_grad()

            fake_img = generator(label_img)
            loss_gan = criterion1(discriminator(fake_img, label_img), real)
            loss_pixbypix = criterion2(fake_img, real_img)

            loss1 = (loss_gan + loss_pixbypix * lambda_criterion2) / 2.0

            loss1.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_real = criterion1(discriminator(real_img, label_img), real)
            loss_fake = criterion1(discriminator(fake_img.detach(), label_img), fake)
            loss2 = (loss_real + loss_fake) / 2.0

            loss2.backward()
            optimizer_d.step()

            done = epoch * len(train_dataloader) + i
            if done % sampling_interval == 0:
                imgs = next(iter(val_dataloader))
                val_real_img, val_label_img = imgs

                val_real_img = val_real_img.cuda()
                val_label_img = val_label_img.cuda()
                val_fake_img = generator(val_label_img)

                sample = torch.cat((val_label_img.data, val_fake_img.data, val_real_img.data), -1)
                save_image(sample, f"{done}.png", nrow=5, normalize=True)

        print(f"[Epoch {epoch}/{epochs}] "
              f"[D loss: {loss1.item():.6f}] "
              f"[G pixel loss: {loss2.item():.6f}, "
              f"adv loss: {loss_gan.item()}]"
              f"[Elapsed time: {time.time() - start_time:.2f}s]")

        # if epoch % 10 == 0:
        #     torch.save(generator.state_dict(), "./pths/pix2pix/Pix2Pix_Generator_Facades" + str(epoch) + ".pth")
        #     torch.save(discriminator.state_dict(), "./pths/pix2pix/Pix2Pix_Discriminator_Facades" + str(epoch) + ".pth")
        #     print("Model saved!")