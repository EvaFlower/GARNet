import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot

import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from tensorboardX import SummaryWriter

from tqdm import tqdm
from read_lfw_dataset import *
from allrank.models.losses import listMLE, listNet
from bable.utils.transforms_utils import ToTensor, MinSizeResize


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
# DATA_DIR = '/data/yinyao/Projects/data/cifar-10/cifar-10-batches-py/'
# if len(DATA_DIR) == 0:
#     raise Exception('Please specify path to data directory in gan_cifar.py!')

DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 1 # How many critic iterations per generator iteration
BATCH_SIZE = 32 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

n_check = 50000
im_size = 32 
img_shape = (im_size, im_size)
use_tensorboard = True
pretrain = False 
d_lr = 2e-4
g_lr = 2e-4
#CLIP_VALUE= 100 
attr_name = 'Smiling'
load_fg = True 
load_lexi = False
score_t = 0.3
n_rank_list = 2 # pairwise ranking with score threshold 0.05
gen_rank_th = 2
d_gt_score_step = 5
g_gt_score_step = d_gt_score_step
LAMBDA0 = 1e-4

output_path = 'results'
run = 1
output_path = os.path.join(output_path, 'adversarial_ranking_{}'.format(attr_name), str(run))
sample_path = os.path.join(output_path, 'samples')
log_path = os.path.join(output_path, 'logs')
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

if use_tensorboard:
    writer = SummaryWriter(log_path)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, im_size, im_size)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)
        #self.linear = nn.Linear(8*8*2*DIM, 1)
        #self.linear = nn.Linear(16*16*DIM, 1)
        #self.linear1 = nn.Linear(16*16*DIM, 100)
        #self.lrelu1 = nn.LeakyReLU()
        #self.linear2 = nn.Linear(100, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.contiguous().view(-1, 4*4*4*DIM)
        #output = output.contiguous().view(-1, 8*8*2*DIM)
        #utput = output.contiguous().view(-1, 16*16*DIM)
        output = self.linear(output)
        #output = self.lrelu1(self.linear1(output))
        #output = self.linear2(output)
        #output = torch.clamp(output, -1*CLIP_VALUE, CLIP_VALUE)
        return output


netG = Generator()
netD = Discriminator()
print(netG)
print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

# optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.9))
# optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

if pretrain:
    load_model_path = '/home/yinyao/Data/Projects/GARNet/adversarial_ranking_utzap/results/wgan-gp_utzap_limited_data/0'
    model_iter = 37999
    #D_file = os.path.join(load_model_path, 'D_{}.pt'.format(model_iter))
    G_file = os.path.join(load_model_path, 'G_{}.pt'.format(model_iter))
    netG.load_state_dict(torch.load(G_file))
    #netD.load_state_dict(torch.load(D_file))


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE*n_rank_list, 1)
    alpha = alpha.expand(BATCH_SIZE*n_rank_list, real_data.nelement()//(BATCH_SIZE*n_rank_list)).contiguous().view(BATCH_SIZE*n_rank_list, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    netG.eval()
    noisev = autograd.Variable(fixed_noise_128)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, os.path.join(sample_path, 'samples_{}.jpg'.format(frame)))


# Dataset iterator
img_shape = (32, 32)
train_data, _ = load_lfw_crop_data(attr=attr_name, img_shape=img_shape)
# load surrogate ranker
ckpt_path = ckpt_path = 'pretrained_ranker/smile.pth'
ranker = torch.load(ckpt_path)
ranker.eval()

# obain psudo label
rank_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.Resize((224, 224)),
    ToTensor(is_rgb=True),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
real_scores = []
n_real = train_data.shape[0]
for k in tqdm(range(n_real//BATCH_SIZE)):
    gen_imgs = train_data[BATCH_SIZE*k: BATCH_SIZE*(k+1)]
    post_imgs = []
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8')
    gen_imgs = torch.stack([rank_preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
if n_real%BATCH_SIZE > 0:
    gen_imgs = train_data[n_real//BATCH_SIZE*BATCH_SIZE: n_real]
    post_imgs = []
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8')
    gen_imgs = torch.stack([rank_preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
    
real_scores = np.concatenate(real_scores, axis=0)
real_scores = np.squeeze(real_scores)
print(real_scores.shape, np.min(real_scores), np.max(real_scores))
train_scores = (real_scores-np.min(real_scores))/(np.max(real_scores)-np.min(real_scores))

# load train data with size 32
# img_shape = (32, 32)
# train_data, _ = load_lfw_crop_data(attr=attr_name, img_shape=(32, 32))

def get_pair_data(batch_size):
    img_pairs = []
    count = 0
    while(count<batch_size):
        idx = np.random.randint(0, train_data.shape[0])
        sample_1 = train_data[idx]
        score_1 = train_scores[idx]
        idx = np.random.randint(0, train_data.shape[0])
        sample_2 = train_data[idx]
        score_2 = train_scores[idx]
        score_diff = score_1-score_2
        if np.abs(score_diff) > score_t:
            if score_diff > 0:
                pair_label = np.array([1, 0])
            else:
                pair_label = np.array([0, 1])
            img_pairs.append((sample_1, sample_2, pair_label))
            count += 1
        else:
            continue
    return img_pairs
            

counter = 0
num_batches = len(train_data)//BATCH_SIZE

for iteration in tqdm(range(ITERS)):
    start_time = time.time()
    # if iteration % 2000 == 0 and iteration>0:
    #     gen_rank_th -= 1
    #     if gen_rank_th < 1:
    #         gen_rank_th = 1
    ############################
    # (1) Update D network
    ###########################
    netD.train()
    netG.train()
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        #_data, _ = next(gen)
        _data = get_pair_data(BATCH_SIZE) #train_data[(counter%num_batches)*BATCH_SIZE:(counter%num_batches+1)*BATCH_SIZE]
        samples_list = []
        for k in range(n_rank_list):
            samples_list.append([d[k] for d in _data])
        list_rank = [d[n_rank_list] for d in _data]

        netD.zero_grad()

        # train with real
        samples_list = np.array(samples_list)
        samples_list = samples_list.reshape(-1, 32, 32, 3)
        samples_list = samples_list.transpose(0, 3, 1, 2)
        real_data_v = torch.tensor(samples_list, dtype=torch.float)
        list_rank = np.array(list_rank)
        gen_rank_th_d = n_rank_list-gen_rank_th
        r_true = list_rank
        #r_true[r_true>=gen_rank_th_d] = r_true[r_true>=gen_rank_th_d]+1
        r_true[r_true>=gen_rank_th_d] = r_true[r_true>=gen_rank_th_d]*d_gt_score_step+10
        r_true = np.concatenate([r_true, np.ones((list_rank.shape[0], 1))*gen_rank_th_d], axis=1)
        r_true = torch.tensor(r_true)
        if use_cuda:
            real_data_v = real_data_v.cuda(gpu)
            r_true = r_true.cuda(gpu)
        # t(r_truprine[:5])

        # import torchvision
        D_real = netD(real_data_v)
        r_out = D_real.reshape(-1, BATCH_SIZE)
        r_out = torch.transpose(r_out, 0, 1)
        # loss epsilon
        r_loss_d_reg = torch.nn.MSELoss()(D_real, torch.zeros_like(D_real))
        D_real = D_real.mean()

        # train with fake
        noise = torch.randn(BATCH_SIZE*n_rank_list, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).detach().data)
        inputv = fake
        D_fake = netD(inputv)
        r_out_fake = D_fake
        D_fake = D_fake.mean()
        
        r_pred = torch.cat([r_out, r_out_fake[:BATCH_SIZE]], dim=1)
        r_loss_d = listNet(r_pred, r_true)
        pair_rank = list_rank[:, :2]
        pair_rank = np.argsort(pair_rank, axis=1)
        pair_rank_tensor = torch.tensor(pair_rank)
        if use_cuda:
            pair_rank_tensor = pair_rank_tensor.cuda(gpu)
        r_loss_good = torch.mean(r_pred[:, :2]*(pair_rank_tensor[:, :2]))*2
        r_loss_bad = torch.mean(r_pred[:, :2]*(1-pair_rank_tensor[:, :2]))*2

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        #gradient_penalty.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = r_loss_d + gradient_penalty * LAMBDA + r_loss_d_reg*LAMBDA0
        Wasserstein_D = D_real - D_fake
        D_cost.backward()
        optimizerD.step()
        counter += 1
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    r_out_fake = G
    G = G.mean()
    #G.backward(mone)
    G_cost = -G
    _data = get_pair_data(BATCH_SIZE) #train_data[(counter%num_batches)*BATCH_SIZE:(counter%num_batches+1)*BATCH_SIZE]
    samples_list = []
    for k in range(n_rank_list):
        samples_list.append([d[k] for d in _data])
    list_rank = [d[n_rank_list] for d in _data]

    samples_list = np.array(samples_list)
    samples_list = samples_list.reshape(-1, 32, 32, 3)
    samples_list = samples_list.transpose(0, 3, 1, 2)
    real_data_v = torch.tensor(samples_list, dtype=torch.float)
    list_rank = np.array(list_rank)
    r_true = list_rank
    r_true = np.concatenate([r_true, np.ones((list_rank.shape[0], 1))*n_rank_list], axis=1)
    #r_true = torch.tensor(r_true)
    r_true = torch.tensor(r_true)*g_gt_score_step
    # print(r_true[:5])
    # exit()
    if use_cuda:
        real_data_v = real_data_v.cuda(gpu)
        r_true = r_true.cuda(gpu)

    r_out = netD(real_data_v)
    r_out = r_out.reshape(-1, BATCH_SIZE)
    r_out = torch.transpose(r_out, 0, 1)
    r_pred = torch.cat([r_out, r_out_fake], dim=1)
    r_loss_g = listNet(r_pred, r_true)
    r_loss_g.backward()
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(os.path.join(output_path, 'Time'), time.time() - start_time)
    lib.plot.plot(os.path.join(output_path, 'Train_disc_cost'), D_cost.cpu().data.numpy())
    lib.plot.plot(os.path.join(output_path, 'Train_gen_cost'), G_cost.cpu().data.numpy())
    lib.plot.plot(os.path.join(output_path, 'Wasserstein_distance'), Wasserstein_D.cpu().data.numpy())
    if use_tensorboard:
        writer.add_scalars('d_out', {'real': D_real, 'fake': D_fake, 'good': r_loss_good, 'bad': r_loss_bad}, iteration)
        writer.add_scalars('loss', {'d_loss': (D_fake-D_real), 'r_loss_d':r_loss_d, 'r_loss_g':r_loss_g, 'loss_gp': gradient_penalty}, iteration)

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 10000 == 9999 or iteration == 0:
        netD.eval()
        netG.eval()
        dev_disc_costs = []
        #for images, _ in dev_gen():
        #    images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        #    imgs = torch.stack([preprocess(item) for item in images])

        #    # imgs = preprocess(images)
        #    if use_cuda:
        #        imgs = imgs.cuda(gpu)
        #    imgs_v = autograd.Variable(imgs)

        #    D = netD(imgs_v)
        #    _dev_disc_cost = -D.mean().cpu().data.numpy()
        #    dev_disc_costs.append(_dev_disc_cost)
        #lib.plot.plot(os.path.join(output_path, 'Dev_disc_cost'), np.mean(dev_disc_costs))
        #save model
        torch.save(netG.state_dict(), os.path.join(output_path, 'G_{}.pt'.format(iteration)))
        torch.save(netD.state_dict(), os.path.join(output_path, 'D_{}.pt'.format(iteration)))
        generate_image(iteration, netG)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()
