import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 56})
from matplotlib import style

SMALL_SIZE = 60
MEDIUM_SIZE = 60
BIGGER_SIZE = 60
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import numpy as np
import seaborn as sns

import tflib as lib
import tflib.save_images
import tflib.plot

import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.autograd as autograd

from model import Generator 
import pandas as pd

from read_utzap_dataset import *
import torchvision
from bable.utils.transforms_utils import ToTensor, MinSizeResize
from PIL import Image
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2023)


def store_data(data, file_path):
    f = open(file_path, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

img_shape = (32, 32)
attr_name = 'comfort' 
attr_dict = {'open': 0, 'pointy': 1, 'sporty': 2, 'comfort': 3}

attr_idx = attr_dict[attr_name]
load_lexi = True

# Initialize generator and discriminator
generator = Generator().cuda(device)

# load cls
# load ranker model
ckpt_path = 'pretrained_ranker/{}.pth'.format(attr_name)
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

n_check = 50000
latent_dim = 128
BATCH_SIZE = 10

# load real data
imgs_and_targets, img_paths = load_utzap_fg_pairs(attr_idx=attr_idx, shape=img_shape, return_path=True)
if load_lexi:
    imgs_and_targets += load_utzap_lexi_pairs(attr_name=attr_name, shape=img_shape)
imgs = []
for it in imgs_and_targets:
    imgs.append(it[0])
    imgs.append(it[1])
imgs = np.array(imgs)
print(imgs.shape)
np.random.shuffle(imgs)
datas = imgs[:n_check]


# evaluate real data's score by a surrogate ranker
load_model_path = 'results/adversarial_ranking_{}/0/'.format(attr_name)

model_epochs = np.arange(199999, 200000, 200000)
save_res_path = os.path.join(load_model_path, 'evaluation')
if not os.path.exists(save_res_path):
    os.makedirs(save_res_path)

real_scores = []
n_real = np.min([n_check, datas.shape[0]])
for k in tqdm(range(n_real//BATCH_SIZE+1)):
    if k*BATCH_SIZE == n_real:
        break
    gen_imgs = datas[BATCH_SIZE*k: BATCH_SIZE*(k+1)]
    post_imgs = []
    gen_imgs = (gen_imgs*0.5+0.5)*255
    gen_imgs = gen_imgs.astype('uint8')
    gen_imgs = torch.stack([preprocess(item) for item in gen_imgs]).to("cuda:0")
    score = ranker.ranker(gen_imgs)
    real_scores.append(score.detach().cpu().numpy())
real_scores = np.concatenate(real_scores, axis=0)
real_scores = np.squeeze(real_scores)
print(real_scores.shape, np.mean(real_scores))

np.save(os.path.join(save_res_path, '{}_scores.npy'.format(attr_name)), real_scores)


# evaluate real and gen's score by ranker
top_5s = []
top_1s = []
ax = plt.subplot(111) 
x = np.arange(10)
dicgan_digits_avg = [] 
for e in tqdm(model_epochs):
    D_file = os.path.join(load_model_path, 'D_{}.pt'.format(e))
    G_file = os.path.join(load_model_path, 'G_{}.pt'.format(e))
    generator.load_state_dict(torch.load(G_file))
    gens = []
    gen_scores = []
    for k in tqdm(range(n_check//BATCH_SIZE)):
        noise = torch.randn(BATCH_SIZE, 128)
        noise = noise.cuda(device)
        noisev = Variable(noise, volatile=True)
        gen_imgs = generator(noisev)
        gen_imgs = gen_imgs.view(-1, 3, 32, 32)
        gens.append(gen_imgs.data.cpu().numpy())
        gen_imgs = (gen_imgs*0.5+0.5)*255
        gen_imgs = gen_imgs.data.cpu().numpy()
        gen_imgs = gen_imgs.astype('uint8')
        gen_imgs = gen_imgs.transpose(0, 2, 3, 1)
        gen_imgs = torch.stack([preprocess(item) for item in gen_imgs]).to("cuda:0")
        score = ranker.ranker(gen_imgs)
        gen_scores.append(score.detach().cpu().numpy())
    gens = np.concatenate(gens, axis=0)
    gen_scores = np.concatenate(gen_scores, axis=0)
    gen_scores = np.squeeze(gen_scores)
    print(gen_scores.shape, np.mean(gen_scores))
    np.save(os.path.join(save_res_path, '{}_scores_gen_{}.npy'.format(attr_name, e)), gen_scores)

print('real', real_scores.shape, np.mean(real_scores))
print('gen', gen_scores.shape, np.mean(gen_scores))

digit_type = np.repeat(np.array('Real'), len(real_scores))
df_real = pd.DataFrame(data={'Score':real_scores, 'type':digit_type}, columns=['Score', 'type'])
k = np.repeat(np.array('Generated'), len(gen_scores))
df_gen = pd.DataFrame(data={'Score': gen_scores, 'type': k}, columns=['Score', 'type'])
dfs = pd.concat([df_real, df_gen])
# Plot hist
def remove_legend_title(ax, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = None 
    ax.legend(handles, labels, title=title, frameon=False, loc='upper left', fontsize=20, **kws)
fig, ax = plt.subplots()
sns_plot = sns.kdeplot(data=dfs, x='Score', hue='type', legend=True, common_norm=False, palette='husl', fill=True)

points_x = np.array([np.around(np.mean(real_scores), 2)])
points_y = np.array(0.).repeat(len(points_x))
line = plt.plot(points_x, points_y, 'o', c='red', markersize=8)[0]
line.set_clip_on(False)
plt.text(points_x[0], 0.015, format(points_x[0], '.2f'), ha='center', va='bottom', fontsize=24)
points_x = np.array([np.around(np.mean(gen_scores), 2)])
points_y = np.array(0.).repeat(len(points_x))
line = plt.plot(points_x, points_y, 'o', c='green', markersize=8)[0]
line.set_clip_on(False)
plt.text(points_x[0], 0.003, format(points_x[0], '.2f'), ha='center', va='bottom', fontsize=24)

if attr_name == 'open':
    plt.ylim(0.0, 0.13)
    plt.yticks(np.arange(0, 0.12, 0.05)) # open
elif attr_name == 'comfort':
    plt.ylim(0.0, 0.185)
    plt.yticks(np.arange(0, 0.152, 0.05)) # comfort
elif attr_name == 'sporty':
    plt.ylim(0.0, 0.11)
    plt.yticks(np.arange(0, 0.12, 0.05)) # sporty
remove_legend_title(sns_plot)
plt.savefig(save_res_path+'/score_kde.pdf', bbox_inches='tight')
