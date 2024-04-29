import cv2
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import mat73
import torch
import torchvision
from bable.utils.transforms_utils import ToTensor
import tflib as lib
import tflib.save_images

def read_one_img(file_path, dim):
    file_path = file_path.replace('./', '%2E/')
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) #io.imread(filepath)
    if img is None:
        print(file_path)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb/127.5-1
    img_rgb = np.float32(img_rgb)
    return img_rgb


def load_utzap_images(num_imgs=50025, shape=(256,256)): 
    root_image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-images-square/'
    image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/image-path.mat'
    img_path_list = loadmat(image_path)['imagepath'][:num_imgs]
    shape = shape
    n = len(img_path_list)
    imgs = [read_one_img(root_image_path+str(img_path_list[i][0][0]), shape) \
        for i in tqdm(range(n))]
    return imgs


def load_utzap_pairs(attr_idx=2, shape=(256, 256), load_equal=True):  # the 3rd is the sporty
    root_image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-images-square/'
    label_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/zappos-labels.mat'
    image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/image-path.mat'
    img_path_list = loadmat(image_path)['imagepath']
    shape = shape 
    
    zappos_label = loadmat(label_path)
    zappos_order = zappos_label['mturkOrder']# (1,4), 4 is # of attributes: open,pointy,sporty,comfort
    zappos_order_attr = zappos_order[0][attr_idx]  
    n = zappos_order_attr.shape[0]
    print(n)
    img_pairs = []
    for idx in tqdm(range(n)):
        pair_1 = int(zappos_order_attr[idx][0]-1)
        pair_2 = int(zappos_order_attr[idx][1]-1)
        pair_label = zappos_order_attr[idx][3]
        if pair_label == 2:
            pair_label = np.array([0, 1]) 
        else:
            pair_label = np.array([1, 0])
        img1_path = root_image_path+str(img_path_list[pair_1][0][0])
        img2_path = root_image_path+str(img_path_list[pair_2][0][0])  
        img1 = read_one_img(img1_path, shape)
        img2 = read_one_img(img2_path, shape) 
        img_pairs.append((img1, img2, pair_label))

    if load_equal:
        zappos_equal = zappos_label['mturkEqual'] 
        zappos_equal_attr = zappos_equal[0][attr_idx] 
        n = zappos_equal_attr.shape[0]
        for idx in tqdm(range(n)):
            pair_1 = int(zappos_equal_attr[idx][0]-1)
            pair_2 = int(zappos_equal_attr[idx][1]-1)
            pair_label = np.array([0, 0]) #zappos_equal_attr[idx][3]
            img1_path = root_image_path+str(img_path_list[pair_1][0][0])
            img2_path = root_image_path+str(img_path_list[pair_2][0][0])   
            img1 = read_one_img(img1_path, shape)
            img2 = read_one_img(img2_path, shape) 
            img_pairs.append((img1, img2, pair_label))
    print(len(img_pairs))
    return img_pairs

def load_utzap_fg_pairs(attr_idx=2, shape=(256, 256), load_fg=True, return_path=False):  # the 3rd is the sporty
    root_image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-images-square/'
    label_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/zappos-labels.mat'
    fg_label_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/zappos-labels-fg.mat'
    image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/image-path.mat'
    img_path_list = loadmat(image_path)['imagepath']
    shape = shape

    zappos_label = loadmat(label_path)
    zappos_order = zappos_label['mturkOrder']# (1,4), 4 is # of attributes: open,pointy,sporty,comfort
    zappos_order_attr = zappos_order[0][attr_idx]
    n = zappos_order_attr.shape[0]
    print(n)
    img_pairs = []
    img_pairs_path = []
    for idx in tqdm(range(n)):
        pair_1 = int(zappos_order_attr[idx][0]-1)
        pair_2 = int(zappos_order_attr[idx][1]-1)
        pair_label = zappos_order_attr[idx][3]
        if pair_label == 2:
            pair_label = np.array([0, 1])
        else:
            pair_label = np.array([1, 0])
        img1_path = root_image_path+str(img_path_list[pair_1][0][0])
        img2_path = root_image_path+str(img_path_list[pair_2][0][0])
        img1 = read_one_img(img1_path, shape)
        img2 = read_one_img(img2_path, shape)
        img_pairs.append((img1, img2, pair_label))
        img_pairs_path.append((img_path_list[pair_1][0][0], img_path_list[pair_2][0][0]))

    if load_fg:
        zappos_fg_label = loadmat(fg_label_path)
        zappos_equal = zappos_fg_label['mturkHard']
        zappos_equal_attr = zappos_equal[0][attr_idx]
        n = zappos_equal_attr.shape[0]
        #print(n)
        for idx in tqdm(range(n)):
            pair_1 = int(zappos_equal_attr[idx][0]-1)
            pair_2 = int(zappos_equal_attr[idx][1]-1)
            pair_label = zappos_equal_attr[idx][3]
            if pair_label == 2:
                pair_label = np.array([0, 1])
            else:
                pair_label = np.array([1, 0])
            img1_path = root_image_path+str(img_path_list[pair_1][0][0])
            img2_path = root_image_path+str(img_path_list[pair_2][0][0])
            img1_path = img1_path.replace('./', '%2E/')
            img2_path = img2_path.replace('./', '%2E/')
            img1 = read_one_img(img1_path, shape)
            img2 = read_one_img(img2_path, shape)
            img_pairs.append((img1, img2, pair_label))
            img_pairs_path.append((str(img_path_list[pair_1][0][0]), img_path_list[pair_2][0][0]))
            
    print(len(img_pairs), len(img_pairs_path))
    if return_path:
        return img_pairs, img_pairs_path
    else:
        return img_pairs
                                                     
def load_utzap_lexi_pairs(attr_name='sporty', shape=(256, 256), return_path=False):  # the 3rd is the sporty
    root_image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-images-square/'
    label_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-lexi/zappos-labels-real-lexi.mat'
    image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-data/image-path.mat'
    img_path_list = loadmat(image_path)['imagepath']
    shape = shape

    zappos_label = mat73.loadmat(label_path)
    zappos_order = zappos_label['mturkOrder']# (1,4), 4 is # of attributes: open,pointy,sporty,comfort
    zappos_attr_name = zappos_label['attrNames']# (1,4), 4 is # of attributes: open,pointy,sporty,comfort
    for idx, name in enumerate(zappos_attr_name):
        if name == attr_name:
            attr_idx = idx
    zappos_order_attr = zappos_order[attr_idx]
    n = zappos_order_attr.shape[0]
    print(n)
    img_pairs = []
    img_pairs_path = []
    for idx in tqdm(range(n)):
        pair_1 = int(zappos_order_attr[idx][0]-1)
        pair_2 = int(zappos_order_attr[idx][1]-1)
        pair_label = zappos_order_attr[idx][3]
        if pair_label == 2:
            pair_label = np.array([0, 1])
        else:
            pair_label = np.array([1, 0])
        img1_path = root_image_path+str(img_path_list[pair_1][0][0])
        img2_path = root_image_path+str(img_path_list[pair_2][0][0])
        img1_path = img1_path.replace('./', '%2E/')
        img2_path = img2_path.replace('./', '%2E/')
        img1 = read_one_img(img1_path, shape)
        img2 = read_one_img(img2_path, shape)
        img_pairs.append((img1, img2, pair_label))
        img_pairs_path.append((img_path_list[pair_1][0][0], img_path_list[pair_2][0][0]))

    print(len(img_pairs), len(img_pairs_path))
    if return_path:
        return img_pairs, img_pairs_path
    else:
        return img_pairs

def get_pair_data_idx(train_data, train_scores, train_data_path, batch_size, score_t=0.5):
    img_pairs = []
    count = 0
    while(count<batch_size):
        idx1 = np.random.randint(0, train_data.shape[0])
        sample_1 = train_data[idx1]
        score_1 = train_scores[idx1]
        idx2 = np.random.randint(0, train_data.shape[0])
        sample_2 = train_data[idx2]
        score_2 = train_scores[idx2]
        score_diff = score_1-score_2
        if np.abs(score_diff) > score_t:
            if score_diff > 0:
                pair_label = np.array([1, 0])
            else:
                pair_label = np.array([0, 1])
            img_pairs.append((train_data_path[idx1], train_data_path[idx2], pair_label))
            count += 1
        else:
            continue
    return img_pairs

def create_utzap_ranker_pair(attr_names=['comfort'], img_shape=(32, 32), load_fg=True, load_lexi=True, score_t=0.1):
    # Dataset iterator
    attr_dict = {'open': 0, 'pointy': 1, 'sporty': 2, 'comfort': 3}
    train_data = []
    train_data_path = []
    for attr_name in attr_names:
        attr_idx = attr_dict[attr_name]
        data, path = load_utzap_fg_pairs(attr_idx=attr_idx, shape=img_shape, load_fg=load_fg, return_path=True)
        train_data += data
        train_data_path += path
        if load_lexi:
            data, path = load_utzap_lexi_pairs(attr_name=attr_name, shape=img_shape, return_path=True)
            train_data += data
            train_data_path += path
    print(len(train_data), len(train_data_path))
    imgs = []
    imgs_path = []
    for it in train_data:
        imgs.append(it[0])
        imgs.append(it[1])   
    for it in train_data_path:
        imgs_path.append(it[0])
        imgs_path.append(it[1])
    imgs = np.array(imgs)
    print(imgs.shape, np.min(imgs), np.max(imgs))
    #np.random.shuffle(imgs)
    train_data = imgs
    train_data_path  = imgs_path
    
    ckpt_path = './pretrained_ranker/{}.pth'.format(attr_name)
    ranker = torch.load(ckpt_path)
    ranker.eval()
    rank_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        #torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Resize((224, 224)),
        ToTensor(is_rgb=True),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    real_scores = []
    n_real = train_data.shape[0]
    BATCH_SIZE = 32
    for k in tqdm(range(n_real//BATCH_SIZE)):
        gen_imgs = train_data[BATCH_SIZE*k: BATCH_SIZE*(k+1)]
        gen_imgs = (gen_imgs*0.5+0.5)*255
        gen_imgs = gen_imgs.astype('uint8')
        gen_imgs = torch.stack([rank_preprocess(item) for item in gen_imgs]).to("cuda:0")
        score = ranker.ranker(gen_imgs)
        real_scores.append(score.detach().cpu().numpy())
    print(torch.min(gen_imgs), torch.max(gen_imgs))
    if n_real%BATCH_SIZE > 0:
        gen_imgs = train_data[n_real//BATCH_SIZE*BATCH_SIZE: n_real]
        gen_imgs = (gen_imgs*0.5+0.5)*255
        gen_imgs = gen_imgs.astype('uint8')
        gen_imgs = torch.stack([rank_preprocess(item) for item in gen_imgs]).to("cuda:0")
        score = ranker.ranker(gen_imgs)
        real_scores.append(score.detach().cpu().numpy())
        
    real_scores = np.concatenate(real_scores, axis=0)
    real_scores = np.squeeze(real_scores)
    print(real_scores.shape, np.min(real_scores), np.max(real_scores), np.mean(real_scores))
    train_scores = (real_scores-np.min(real_scores))/(np.max(real_scores)-np.min(real_scores))
    print(train_scores.shape, np.min(train_scores), np.max(train_scores))
    print(np.mean(real_scores[train_scores>0.5]), np.sum(train_scores<0.5), np.sum(train_scores==0.5))
    img_pairs = get_pair_data_idx(train_data, train_scores, train_data_path, 50000, score_t=score_t)
    np.save('img_pairs_{}_{}.npy'.format(attr_name, score_t), img_pairs)
    img_pairs = np.load('img_pairs_{}_{}.npy'.format(attr_name, score_t), allow_pickle=True)
    print(len(img_pairs))

def read_utzap_ranker_pair(attr_name, shape=(64, 64), score_t=0.5):
    root_image_path = '/data/yinghua/datasets/UTZAP50K/ut-zap50k-images-square/'
    img_pairs_path = np.load('img_pairs_{}_{}.npy'.format(attr_name, score_t), allow_pickle=True)
    img_pairs = []
    for it in tqdm(img_pairs_path[:10]):
        img1_path = root_image_path+str(it[0])
        img2_path = root_image_path+str(it[1])
        img1 = read_one_img(img1_path, shape)
        img2 = read_one_img(img2_path, shape)
        img_pairs.append((img1, img2, it[2]))
    print(len(img_pairs), img_pairs[0][0].shape, img_pairs[0][2], np.max(img_pairs[0][0]), np.min(img_pairs[0][0]))
    return img_pairs
        


if __name__ == '__main__':
    img_shape = (32, 32)
    load_utzap_fg_pairs()

