import torch
import torchvision
from PIL import Image
from bable.utils.transforms_utils import ToTensor, MinSizeResize
import numpy as np
from torchvision import transforms


class BasePredictDataset(torch.utils.data.Dataset):
    def __init__(self, min_height=224, min_width=224, is_bgr=False):
        super(BasePredictDataset, self).__init__()
        self._is_bgr = is_bgr
        self._min_height = min_height
        self._min_width = min_width
        self._image_full_paths = self._get_image_full_paths()
        self._transforms = self._get_transforms()

    def _get_image_full_paths(self):
        raise NotImplementedError

    def _get_transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.Resize((self._min_height, self._min_width)),
            #MinSizeResize(self._min_height, self._min_width),
            ToTensor(is_rgb=not self._is_bgr),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __getitem__(self, index):
        img = Image.open(self._image_full_paths[index])
        #print(np.asarray(img).shape)
        #print(np.min(np.array(img)), np.max(np.array(img)))
        img = self._transforms(img)
        #print(torch.min(img), torch.max(img))
        #print(np.asarray(img).shape)
        return self._image_full_paths[index], img

    def __len__(self):
        return len(self._image_full_paths)


class BaseSiameseDataset(torch.utils.data.Dataset):
    def __init__(self, split, category_id, transforms_config):
        super(BaseSiameseDataset, self).__init__()
        assert split in self._get_splits(), 'unknown split %s' % split
        assert category_id in range(len(self._get_categories())), \
            'invalid category_id %d ' % category_id

        self._category_name = self._get_categories()[category_id]
        self._list, self._labels = self._get_list_and_labels(split)
        self._trans_config = transforms_config
        self._transforms = self._get_transforms()

    @property
    def category_name(self):
        return self._category_name

    def _get_splits(self):
        raise NotImplementedError

    def _get_categories(self):
        raise NotImplementedError

    def _get_list_and_labels(self, split):
        raise NotImplementedError

    def _get_transforms(self):
        transforms_list = []

        if self._trans_config.get('init_size'):
            transforms_list.append(
                torchvision.transforms.Resize(
                    self._trans_config.get('init_size')
                )
            )

        if self._trans_config.get('color_jitter'):
            transforms_list.append(torchvision.transforms.ColorJitter(
                brightness=self._trans_config.get('brightness'),
                contrast=self._trans_config.get('contrast'),
                saturation=self._trans_config.get('saturation'),
                hue=self._trans_config.get('hue'),
            ))

        if self._trans_config.get('random_horizontal_flip'):
            transforms_list.append(
                torchvision.transforms.RandomHorizontalFlip()
            )

        if self._trans_config.get('resize_size'):
            transforms_list.append(
                torchvision.transforms.Resize(
                    self._trans_config.get('resize_size')
                )
            )

        if self._trans_config.get('random_resized_crop_size'):
            transforms_list.append(
                torchvision.transforms.RandomResizedCrop(
                    self._trans_config.get('random_resized_crop_size'),
                    self._trans_config.get('random_resized_crop_scale'),
                    self._trans_config.get('random_resized_crop_ratio'),
                )
            )

        transforms_list.append(
            ToTensor(is_rgb=self._trans_config.get('is_rgb'))
        )

        if self._trans_config.get('normalize'):
            transforms_list.append(self._trans_config['normalize'])

        return torchvision.transforms.Compose(transforms_list)

    def __getitem__(self, index):
        label = self._labels[index]
        img_p1, img_p2 = self._list[index]
        img1 = Image.open(img_p1)
        img2 = Image.open(img_p2)
        print(np.min(np.array(img1)), np.max(np.array(img1)))
        img1 = self._transforms(img1)
        img2 = self._transforms(img2)
        print(torch.min(img1), torch.max(img1))
        return (img1, img2), torch.tensor(label)

    def __len__(self):
        return len(self._labels)
