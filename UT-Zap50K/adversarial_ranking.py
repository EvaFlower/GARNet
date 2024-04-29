import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.plot

import numpy as np
import random

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from tensorboardX import SummaryWriter

from tqdm import tqdm
from read_utzap_dataset import *
from allrank.models.losses import listMLE, listNet
from bable.utils.transforms_utils import ToTensor, MinSizeResize

from model import Generator, Adv_Ranker


# SEED = 2022
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministics = True
# torch.backends.cudnn.benchmark = False

DIM = 128 
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 1 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for

n_check = 50000
im_size = 32 
img_shape = (im_size, im_size)
use_tensorboard = True
d_lr = 2e-4
g_lr = 2e-4
attr_name = 'sporty'
attr_dict = {'open': 0, 'pointy': 1, 'sporty': 2, 'comfort': 3}
attr_idx = attr_dict[attr_name]
load_fg = True
load_lexi = True 
d_gt_score_step = 5
g_gt_score_step = d_gt_score_step
LAMBDA0 = 1e-4

output_path = 'results'
run = '0' 
output_path = os.path.join(output_path, 'adversarial_ranking_{}'.format(\
    attr_name), str(run))
sample_path = os.path.join(output_path, 'samples')
log_path = os.path.join(output_path, 'logs')
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

if use_tensorboard:
    writer = SummaryWriter(log_path)


netG = Generator()
netR = Adv_Ranker()
print(netG)
print(netR)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netR = netR.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netR.parameters(), lr=d_lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE*2, 1)
    alpha = alpha.expand(BATCH_SIZE*2, real_data.nelement()//(BATCH_SIZE*2)).contiguous().view(BATCH_SIZE*2, 3, 32, 32)
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
train_data = load_utzap_fg_pairs(attr_idx=attr_idx, shape=img_shape, load_fg=load_fg)
if load_lexi:
    train_data += load_utzap_lexi_pairs(attr_name=attr_name, shape=img_shape)
print(len(train_data))
imgs = []
for it in train_data:
    imgs.append(it[0])
    imgs.append(it[1])
imgs = np.array(imgs)
print(imgs.shape)
np.random.shuffle(imgs)
train_data = imgs

# load surrogate ranker
ckpt_path = 'pretrained_ranker/{}.pth'.format(attr_name)
ranker = torch.load(ckpt_path)
ranker.eval()

ranker_preprocess = torchvision.transforms.Compose([
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
    gen_imgs = torch.stack([ranker_preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
if n_real%BATCH_SIZE > 0:
    gen_imgs = train_data[n_real//BATCH_SIZE*BATCH_SIZE: n_real]
    post_imgs = []
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8')
    gen_imgs = torch.stack([ranker_preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
    
real_scores = np.concatenate(real_scores, axis=0)
real_scores = np.squeeze(real_scores)
print(real_scores.shape, np.min(real_scores), np.max(real_scores))
train_scores = (real_scores-np.min(real_scores))/(np.max(real_scores)-np.min(real_scores))

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
        if np.abs(score_diff) > 0.1:
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
    ############################
    # (1) Update D network
    ###########################
    netR.train()
    netG.train()
    for p in netR.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        _data = get_pair_data(BATCH_SIZE) 
        samples_1 = [d[0] for d in _data]
        samples_2 = [d[1] for d in _data]
        pair_rank = [d[2] for d in _data]
        netR.zero_grad()

        # train with real
        samples_1 = np.array(samples_1)
        samples_1 = samples_1.transpose(0, 3, 1, 2) 
        samples_2 = np.array(samples_2)
        samples_2 = samples_2.transpose(0, 3, 1, 2) 
        samples_1 = torch.tensor(samples_1) 
        samples_2 = torch.tensor(samples_2) 
        r_true = np.array(pair_rank)
        r_true = r_true*d_gt_score_step+10
        r_true = np.concatenate([r_true, np.zeros((r_true.shape[0], 1))], axis=1)
        r_true = torch.tensor(r_true)

        if use_cuda:
            samples_1 = samples_1.cuda(gpu)
            samples_2 = samples_2.cuda(gpu)
            r_true = r_true.cuda(gpu)
        real_data_v = torch.cat([samples_1, samples_2], dim=0)

        D_real = netR(real_data_v)
        r_loss_d_reg = torch.nn.MSELoss()(D_real, torch.zeros_like(D_real))
        D_real = D_real.mean()

        # train with fake
        noise = torch.randn(BATCH_SIZE*2, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).detach().data)
        inputv = fake
        D_fake = netR(inputv)
        r_out_fake = D_fake
        D_fake = D_fake.mean()

        r_out_first = netR(samples_1)
        r_out_second = netR(samples_2)
        r_pred = torch.cat([r_out_first, r_out_second, r_out_fake[:BATCH_SIZE]], dim=1)
        r_loss_d = listNet(r_pred, r_true)
        pair_rank_tensor = torch.tensor(pair_rank)
        if use_cuda:
            pair_rank_tensor = pair_rank_tensor.cuda(gpu)
        r_loss_good = torch.mean(r_pred[:, :2]*(pair_rank_tensor[:, :2]))*2
        r_loss_bad = torch.mean(r_pred[:, :2]*(1-pair_rank_tensor[:, :2]))*2

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netR, real_data_v.data, fake.data)

        D_cost = r_loss_d + gradient_penalty * LAMBDA + r_loss_d_reg * LAMBDA0
        Wasserstein_D = D_real - D_fake
        D_cost.backward()
        optimizerD.step()
        counter += 1
    ############################
    # (2) Update G network
    ###########################
    for p in netR.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netR(fake)
    r_out_fake = G
    G = G.mean()
    G_cost = -G
    _data = get_pair_data(BATCH_SIZE) 
    samples_1 = [d[0] for d in _data]
    samples_2 = [d[1] for d in _data]
    pair_rank = [d[2] for d in _data]

    # train with real
    samples_1 = np.array(samples_1)
    samples_1 = samples_1.transpose(0, 3, 1, 2) 
    samples_2 = np.array(samples_2)
    samples_2 = samples_2.transpose(0, 3, 1, 2) 
    samples_1 = torch.tensor(samples_1) 
    samples_2 = torch.tensor(samples_2) 
    pair_rank = np.array(pair_rank)
    r_true = np.concatenate([pair_rank, np.ones((BATCH_SIZE, 1))*2], axis=1)
    r_true = torch.tensor(r_true)*g_gt_score_step

    if use_cuda:
        samples_1 = samples_1.cuda(gpu)
        samples_2 = samples_2.cuda(gpu)
        r_true = r_true.cuda(gpu)
    r_out_first = netR(samples_1)
    r_out_second = netR(samples_2)
    r_pred = torch.cat([r_out_first, r_out_second, r_out_fake[:BATCH_SIZE]], dim=1)
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
        netR.eval()
        netG.eval()
        dev_disc_costs = []
        torch.save(netG.state_dict(), os.path.join(output_path, 'G_{}.pt'.format(iteration)))
        torch.save(netR.state_dict(), os.path.join(output_path, 'D_{}.pt'.format(iteration)))
        generate_image(iteration, netG)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()
