import cv2
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import h5py
import mat73


def read_one_img(file_path, dim):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) #io.imread(filepath)
    if img is None:
        print(file_path)
    #cv2.imwrite('ori.jpg', img)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    #cv2.imwrite('ori_256.jpg', img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb/127.5-1
    #print(np.max(img_rgb), np.min(img_rgb))
    #print(np.max(img_rgb), np.min(img_rgb))
    #image = (img_rgb/2+0.5)*255
    #image = image.astype(np.uint8)
    #img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('tmp.jpg', img_bgr)
    return img_rgb


def load_images(img_path_list, shape, offset, num_imgs, root_image_path):
    n = len(img_path_list)
    imgs = [read_one_img(root_image_path+str(img_path_list[i]).replace('\\', '/'), shape) \
        for i in tqdm(range(offset, offset+num_imgs))]
    print(len(imgs))
    return imgs


def load_lfw_data(attr='Smiling', img_shape=(256,256), offset=0, num_imgs=13143):
    anno_file_path = '/data/yinghua/datasets/LFW/LFW/lfw_att_73.mat'
    root_image_path = '/data/yinghua/datasets/LFW/LFW/lfw/'
    annos = mat73.loadmat(anno_file_path)
    img_path_list = annos['name']
    attr_name = np.array(annos['AttrName'])
    #print(attr_name)
    imgs = load_images(img_path_list, img_shape, offset, num_imgs, root_image_path)
    imgs = np.array(imgs)
    attrs = np.array(annos['label'])
    #for an in attr_name:
    #    attr_idx = np.where(attr_name==an)[0]
    #    attr_ = attrs[:, attr_idx]
    #    print(an, np.sum(attr_==1), np.sum(attr_==0))
    attr_idx = np.where(attr_name==attr)[0]
    print(img_path_list[:10])
    return imgs, attrs[offset:offset+num_imgs, attr_idx]
    

def load_lfw_crop_data(attr='Smiling', img_shape=(256,256), offset=0, num_imgs=13143):
    anno_file_path = '/data/yinghua/datasets/LFW/LFW/lfw_att_73.mat'
    root_image_path = '/data/yinghua/datasets/LFW/LFW/lfw_crop/'
    annos = mat73.loadmat(anno_file_path)
    img_path_list = annos['name']
    attr_name = np.array(annos['AttrName'])
    #print(attr_name)
    imgs = load_images(img_path_list, img_shape, offset, num_imgs, root_image_path)
    imgs = np.array(imgs)
    attrs = np.array(annos['label'])
    #for an in attr_name:
    #    attr_idx = np.where(attr_name==an)[0]
    #    attr_ = attrs[:, attr_idx]
    #    print(an, np.sum(attr_==1), np.sum(attr_==0))
    attr_idx = np.where(attr_name==attr)[0]
    return imgs, attrs[offset:offset+num_imgs, attr_idx]
    

def load_lfw_crop_top_images(attr_name='smile', img_shape=(32, 32), top_ratio=0.5, dataset_type='lfw_crop_predict'):
    root_path = '/home/yinyao/Data/Projects/GARNet/attribute_ranking/pytorch-relative-attributes/'
    paths = np.load(root_path+'{}_results/{}_top{}_rank_image_path.npy'.format(dataset_type, attr_name, top_ratio))
    imgs = [read_one_img(p, img_shape) for p in tqdm(paths)]
    imgs = np.array(imgs)
    print(paths[:10])
    print(imgs.shape)
    return imgs


if __name__ == '__main__':
    imgs, attrs = load_lfw_data('Bald')
    print(np.sum(attrs==1), np.sum(attrs==0))
    #print(imgs.shape, attrs.shape)
    print(imgs.shape, attrs.shape, np.max(imgs[0]), np.min(imgs[1]))

