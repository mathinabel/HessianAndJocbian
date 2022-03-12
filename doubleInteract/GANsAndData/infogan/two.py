import itertools
import os
import numpy as np
import math
import itertools

from torch.autograd import Variable

import torch.nn as nn

from tqdm import tqdm
import torch
import torchvision as tv
from torch.utils.data import DataLoader


class Config(object):
    data_path = 'G:/pycharm proj\GANerciyuan\one\data/'
    virs = "result"
    num_workers = 4  # 多线程
    img_size = 96  # 剪切图片的像素大小
    # img_size = 384
    batch_size = 256  # 批处理数量
    max_epoch = 1000  # 最大轮次
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    lr = 0.0002
    beta1 = 0.5  # 正则化系数，Adam优化器参数
    gpu = True  # 是否使用GPU运算（建议使用）
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的卷积核个数
    ndf = 64  # 判别器的卷积核个数
    # 1.模型保存路径
    save_path = '../dcgan/imgs2/'  # opt.netg_path生成图片的保存路径
    # 判别模型的更新频率要高于生成模型
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每1个batch训练一次生成模型
    save_every = 1  # 每save_every次保存一次模型
    netd_path = None
    netg_path = None
    # 测试数据
    gen_img = "result.png"
    # 选择保存的照片
    # 一次生成保存64张图片
    gen_num = 1
    gen_search_num = 512
    gen_mean = 0  # 生成模型的噪声均值
    gen_std = 1  # 噪声方差
    n_classes = 10
    latent_dim = 62
    code_dim = 2


# 实例化Config类，设定超参数，并设置为全局参数
opt = Config()
cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim
        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size** 2 )
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
 # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
 # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


def train(**kwargs):
    # 配置属性
    # 如果函数无字典输入则使用opt中设定好的默认超参数
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # device(设备)，分配设备
    if opt.gpu:
        #       device = torch.device('cpu')
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    # 数据预处理1
    # transforms 模块提供一般图像转换操作类的功能，最后转成floatTensor
    # tv.transforms.Compose用于组合多个tv.transforms操作,定义好transforms组合操作后，直接传入图片即可进行处理
    # tv.transforms.Resize，对PIL Image对象作resize运算， 数值保存类型为float64
    # tv.transforms.CenterCrop, 中心裁剪
    # tv.transforms.ToTensor，将opencv读到的图片转为torch image类型（通道，像素，像素）,且把像素范围转为[0，1]
    # tv.transforms.Normalize,执行image = (image - mean)/std 数据归一化操作，一参数是mean,二参数std
    # 因为是三通道，所以mean = (0.5, 0.5, 0.5),从而转成[-1, 1]范围
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.img_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(root=opt.data_path, transform=transforms)
    dataloader = DataLoader(
        dataset,  # 数据加载
        batch_size=opt.batch_size,  # 批处理大小设置
        shuffle=True,  # 是否进行洗牌操作
        num_workers=opt.num_workers,  # 是否进行多线程加载数据设置
        drop_last=True  # 为True时，如果数据集大小不能被批处理大小整除，则设置为删除最后一个不完整的批处理。
    )
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()
    # 初始化网络
    generator = Generator()
    discriminator = Discriminator()
    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1
    # 定义标签，并且开始注入生成器的输入noise
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    # 生成满足N(1,1)标准正态分布，opt.nz维（100维），opt.batch_size个数的随机噪声
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    # 用于保存模型时作生成图像示例
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()
    map_location = lambda storage, loc: storage
    optimize_g = torch.optim.Adam(generator.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimize_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    criterions = nn.BCELoss().to(device)
    for epoch in range(opt.max_epoch):
        for ii_, (img, _) in tqdm((enumerate(dataloader))):
            real_img = img.to(device)
            if ii_ % opt.d_every == 0:
                optimize_d.zero_grad()
                # 训练判别器
                # 把判别器的目标函数分成两段分别进行反向求导，再统一优化
                output, _, _ = discriminator(real_img)
                error_d_real = adversarial_loss(output, true_labels)
                noises = noises.detach()
                z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                label_input = to_categorical(np.random.randint(0, opt.n_classes, opt.batch_size), num_columns=opt.n_classes)
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (opt.batch_size, opt.code_dim))))

                fake_image = generator(noises,label_input, code_input).detach()
                output, _, _  = discriminator(fake_image)
                error_d_fake = adversarial_loss(output, fake_labels)
                d_loss = (error_d_real + error_d_fake) / 2
                d_loss.backward()
                optimize_d.step()
            # 训练生成器
            if ii_ % opt.g_every == 0:
                optimize_g.zero_grad()
                # 用于netd作判别训练和用于netg作生成训练两组噪声需不同
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_image = generator(noises,label_input, code_input)
                output = discriminator(fake_image)
                # 此时判别器已经固定住了，BCE的一项为定值，再求最小化相当于求二项即G得分的最大化
                error_g = adversarial_loss(output, true_labels)
                error_g.backward()
                # 计算一次Adam算法，完成判别模型的参数迭代
                optimize_g.step()
            if ii_ % opt.d_every == 0:
                # ------------------
                # Information Loss
                # ------------------

                optimizer_info.zero_grad()

                # Sample labels
                sampled_labels = np.random.randint(0, opt.n_classes, opt.batch_size)

                # Ground truth labels
                gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

                # Sample noise, labels and code as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
                label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
                code_input = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (opt.batch_size, opt.code_dim))))

                gen_imgs = generator(z, label_input, code_input)
                _, pred_label, pred_code = discriminator(gen_imgs)

                info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                    pred_code, code_input
                )
                info_loss.backward()
                optimizer_info.step()

        # 保存模型
        if (epoch + 1) % opt.save_every == 0:
            fix_fake_image = generator(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:1], "%s/%s.png" % (opt.save_path, epoch), normalize=True)

            torch.save(discriminator.state_dict(), 'imgs2/' + 'netd_{0}.pth'.format(epoch))
            torch.save(generator.state_dict(), 'imgs2/' + 'netg_{0}.pth'.format(epoch))


# @torch.no_grad():数据不需要计算梯度，也不会进行反向传播
@torch.no_grad()
def generate(**kwargs):
    # 用训练好的模型来生成图片

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda") if opt.gpu else torch.device("cpu")

    # 加载训练好的权重数据
    netg, netd = Generator().eval(), Discriminator().eval()
    #  两个参数返回第一个
    map_location = lambda storage, loc: storage

    # opt.netd_path等参数有待修改
    netd.load_state_dict(torch.load('imgs2/netd_399.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('imgs2/netg_399.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    # 生成训练好的图片
    # 初始化512组噪声，选其中好的拿来保存输出。
    noise = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)

    fake_image = netg(noise)
    score = netd(fake_image).detach()

    # 挑选出合适的图片
    # 取出得分最高的图片
    indexs = score.topk(opt.gen_num)[1]

    result = []

    for ii in indexs:
        result.append(fake_image.data[ii])

    # 以opt.gen_img为文件名保存生成图片
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


def main():
    # 训练模型
    train()
    # 生成图片
    generate()


if __name__ == '__main__':
    main()
