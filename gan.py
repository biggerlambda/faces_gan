import progressbar
import torch
import argparse
import os
from torch import nn
import torchvision.datasets as dset
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.path.expanduser("~/datasets/celebA/Img_align_celeba"), help="root image folder")
parser.add_argument("--imageSize", type=int, default=64, help="image size")
parser.add_argument("--batchSize", type=int, default=64, help="batch size")
parser.add_argument("--nz", type=int, default=100, help="size of input noise vector")
parser.add_argument("--nepochs", type=int, default=25, help="number of epochs")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--output_dir", default=os.path.expanduser("."), help="output directory")
parser.add_argument("--workers", type=int, default=2, help="number of data loaders")
parser.add_argument("--visdom", type=bool, default=True, help="enable visdom")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

args = parser.parse_args()
print(args)

# visdom stuff
if args.visdom:
    from visdom import Visdom
    vis = Visdom()

dataset = dset.ImageFolder(root=args.root, 
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.CenterCrop(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose2d(args.nz, 1024, 4, 1, 0, bias=False),
                # output is 1024 filters of 4 x 4
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                # output is 512 filters of 8x 8 
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                # output is 256 filters of 16 x 16 
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                # output is 128 filters of 32 x 32 
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                # output is 3 filters of 64 x 64
                nn.Tanh()
                )

    def forward(self, input):
        return self.model(input)


class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        # input is batch x 3 x 64 x 64
        self.model = nn.Sequential(
                nn.Conv2d(3, 128, 4, 2, 1, bias=False),
                # output is 128 filters of 32 x 32
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                # output is 256 filters of 16 x 16
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                # output is 512 filters of 8 x 8
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                # output is 1024 filters of 4x4
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, input):
        x = self.model(input)
        return x.view(-1, 1)

gen_model = Gen().cuda()
disc_model = Disc().cuda()

# define nll loss for discriminator
bce_criterion = nn.BCELoss().cuda()

# set the optimizer for both generative and discriminative model
gen_optim = optim.Adam(gen_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(disc_model.parameters(), lr=args.lr, betas=(0.5, 0.999))

torch.cuda.manual_seed(args.seed)

# set models on train mode
gen_model.train()
disc_model.train()

# fake and real label indices for discriminator output
real_label = torch.Tensor([0]).cuda()
fake_label = torch.Tensor([1]).cuda()

# train on epochs
for i in range(args.nepochs):
    bar = progressbar.ProgressBar(max_value = len(dataset.imgs))
    j = 0
    for batch_idx, data_tup in enumerate(dataloader):
        # data and label
        data, _ = data_tup
        batch_size = data.size(0)
        data = data.cuda()
        
        j += batch_size
        bar.update(j)
        # train the discriminator
        # real data
        disc_model.zero_grad()
        var_data = Variable(data)
        output = disc_model(var_data)
        labels = Variable(real_label.expand_as(output))
        disc_loss = bce_criterion(output, labels)

        disc_loss.backward()
        disc_optim.step()

        # fake data
        disc_model.zero_grad()
        z = Variable(torch.randn(data.size(0),args.nz, 1, 1).cuda())
        fake_data = gen_model(z)
        output = disc_model(fake_data)
        labels = Variable(fake_label.expand_as(output))
        disc_loss = bce_criterion(output, labels)
        disc_loss.backward()
        disc_optim.step()

        # train the generator. maximize log(D(G(x)))
        gen_model.zero_grad()
        z = Variable(torch.randn(data.size(0), args.nz, 1, 1).cuda())
        fake_data = gen_model(z)
        output = disc_model(fake_data)
        labels = Variable(real_label.expand_as(output))
        gen_loss = bce_criterion(output, labels)
        gen_loss.backward()
        gen_optim.step()

        # generate an image, every 100 steps.
        if batch_idx % 100 == 0:
            if args.visdom:
                imgtensor = fake_data.data.cpu()
                imgtensor.ndim = 4
                grid_tensor = vutils.make_grid(imgtensor)
                vis.image(grid_tensor)
            else:
                vutils.save_image(fake_data.data, "{0}/gen_image.png".format(args.output_dir), normalize=True)


    #save the model
    torch.save(gen_model.state_dict(), "{0}/gen_model_epoch_{1}".format(args.output_dir, i))
    torch.save(disc_model.state_dict(), "{0}/disc_model_epoch_{1}".format(args.output_dir, i))
