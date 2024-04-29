
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from model import Generator
import tflib as lib
import fid
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from read_utzap_dataset import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

epoch = 199999
BATCH_SIZE = 50
n_check = 50000

img_shape = (32, 32)
attr_name = 'comfort' 
attr_dict = {'open': 0, 'pointy': 1, 'sporty': 2, 'comfort': 3}

# load data
attr_idx = attr_dict[attr_name]
load_lexi = True
imgs_and_targets = load_utzap_fg_pairs(attr_idx=attr_idx, shape=img_shape)
if load_lexi:
    tmps = load_utzap_lexi_pairs(attr_name=attr_name, shape=img_shape)
    imgs_and_targets += tmps
imgs = []
for it in imgs_and_targets:
    imgs.append(it[0])
    imgs.append(it[1])
data = np.array(imgs)

# select top 50% as desired data
ckpt_path = './pretrained_ranker/{}.pth'.format(attr_name)
ranker = torch.load(ckpt_path)
ranker.eval()
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.Resize((224, 224)),
    ToTensor(is_rgb=True),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),])
real_scores = []
n_real = np.min([n_check, data.shape[0]])
for k in tqdm(range(n_real//BATCH_SIZE+1)):
    if k*BATCH_SIZE == n_real:
        break
    gen_imgs = data[BATCH_SIZE*k: BATCH_SIZE*(k+1)]
    post_imgs = []
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8')
    gen_imgs = torch.stack([preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
real_scores = np.concatenate(real_scores, axis=0)
real_scores = np.squeeze(real_scores)
sort_idx = np.argsort(real_scores)
ids = sort_idx[-int(len(sort_idx)*0.5):]
data = data[ids]

# obtain generative data
load_model_path = './results/adversarial_ranking_{}/0/'.format(attr_name)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
generator = Generator().cuda(device)
G_file = os.path.join(load_model_path, 'G_{}.pt'.format(epoch))
generator.load_state_dict(torch.load(G_file))
gen_data = []
save_imgs = []
for k in tqdm(range(n_check//BATCH_SIZE)):
    noise = torch.randn(BATCH_SIZE, 128)
    noise = noise.cuda(device)
    noisev = Variable(noise, volatile=True)
    gen_imgs = generator(noisev)
    gen_imgs = gen_imgs.view(-1, 3, 32, 32)
    save_imgs = gen_imgs
    gen_imgs = gen_imgs.data.cpu().numpy()
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8').transpose(0, 2, 3, 1)
    gen_data.append(gen_imgs)
gen_data = np.concatenate(gen_data, axis=0)
    
# calculate fid
real_data = ((data+1)/2*255).astype(np.uint8)
real_data = np.concatenate([real_data, real_data], axis=0)
print(real_data.shape, gen_data.shape)
inception_path = fid.check_or_download_inception(None) # download inception network
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(gen_data, sess, batch_size=100)
    mu_real, sigma_real = fid.calculate_activation_statistics(real_data[:n_check], sess, batch_size=100) 
fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)

