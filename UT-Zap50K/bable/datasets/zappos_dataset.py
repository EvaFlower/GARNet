import platform
import os
import numpy as np
import scipy.io as sio
from bable.datasets.base_dataset import BaseSiameseDataset, BasePredictDataset
from bable.utils.transforms_utils import get_default_transforms_config
import mat73
import glob


if 'Windows' in platform.platform():
    BASE_DATASET = "F:\\data\\zap50k"
else:
    BASE_DATASET = "/hdd02/zhangyiyang/data/zap50k"

BASE_DATASET = '/data/yinyao/Projects/data/utzap50k'

SPLITS = ('train', 'test')
CATEGORIES = ('open', 'pointy', 'sporty', 'comfort')


def _get_image_names(image_names_raw, image_dir):
    # 下载下来的数据集路径存在问题，需要修正……
    # 改写 DRN 中的代码
    # you see this crazy for loop? yes I hate it too.
    image_names = []
    for p in image_names_raw:
        this_thing = str(p[0])
        this_thing_parts = this_thing.rsplit('/', 1)
        if this_thing_parts[0].endswith('.'):
            this_thing_parts[0] = this_thing_parts[0][:-1]
            this_thing = '/'.join(this_thing_parts)

        if image_dir.endswith('square'):
            if "Aquatalia by Marvin K" in this_thing_parts[0]:
                this_thing_parts[0] = this_thing_parts[0].replace(
                    "Aquatalia by Marvin K", "Aquatalia by Marvin K%2E")
                this_thing = '/'.join(this_thing_parts)
            elif "Neil M" in this_thing_parts[0]:
                this_thing_parts[0] = this_thing_parts[0].replace(
                    "Neil M", "Neil M%2E")
                this_thing = '/'.join(this_thing_parts)
            elif "W.A.G" in this_thing_parts[0]:
                this_thing_parts[0] = this_thing_parts[0].replace(
                    "W.A.G", "W.A.G%2E")
                this_thing = '/'.join(this_thing_parts)
            elif "L.A.M.B" in this_thing_parts[0]:
                this_thing_parts[0] = this_thing_parts[0].replace(
                    "L.A.M.B", "L.A.M.B%2E")
                this_thing = '/'.join(this_thing_parts)
        else:
            if "Levi's" in this_thing_parts[0]:
                this_thing_parts[0] = this_thing_parts[0].replace(
                    "Levi's", "Levi's&#174;")
                this_thing = '/'.join(this_thing_parts)

        image_names.append(os.path.join(image_dir, this_thing))
    return image_names


class ZapposV1(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 pairs_file_name="train-test-splits-pairs.mat",
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._pairs_file_name = pairs_file_name

        super(ZapposV1, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):

        def _convert_winnter(old):
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
            if old == 3:
                return 0
            if old == 1:
                return -1
            return 1

        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        pairs_file = sio.loadmat(opj(annotation_dir, self._pairs_file_name))
        if split == 'train':
            ndarray = pairs_file['trainPairsAll'].flatten()[
                self._category_id].flatten()[0]
        else:
            ndarray = pairs_file['testPairsAll'].flatten()[
                self._category_id].flatten()[0]
        ndarray = ndarray.astype(np.int32)
        if not self._include_equal:
            ids = np.where(ndarray[:, 3] != 3)[0]
            ndarray = ndarray[ids]

        # list
        pair_list = [(image_names[ndarray[idx, 0] - 1],  # 注意，要-1
                      image_names[ndarray[idx, 1] - 1])  # 注意，要-1
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels


class ZapposV2(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name

        super(ZapposV2, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):
        def _convert_winnter(old):
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
            if old == 3:
                return 0
            if old == 1:
                return -1
            return 1

        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        if split == 'train':
            ndarray = sio.loadmat(
                opj(annotation_dir, self._labels_file_name)
            )['mturkOrder'].flatten()[self._category_id]
            if self._include_equal:
                ndarray2 = sio.loadmat(
                    os.path.join(annotation_dir, self._labels_file_name)
                )['mturkEqual'].flatten()[self._category_id]
                ndarray = np.concatenate([ndarray2, ndarray])
        else:
            ndarray = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]

        # list
        ndarray = ndarray.astype(np.int32)
        pair_list = [(image_names[ndarray[idx, 0] - 1],  # 注意，要-1
                      image_names[ndarray[idx, 1] - 1])  # 注意，要-1
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels


class ZapposV3(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=True,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name

        super(ZapposV3, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):
        def _convert_winnter(old):
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
            if old == 3:
                return 0
            if old == 1:
                return -1
            return 1

        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        ndarray = sio.loadmat(
            opj(annotation_dir, self._labels_file_name)
        )['mturkOrder'].flatten()[self._category_id]
        if self._include_equal:
            ndarray2 = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]
            ndarray = np.concatenate([ndarray2, ndarray])
        split_ratio = 0.9
        train_size = int(len(ndarray)*split_ratio)
        np.random.shuffle(ndarray)
        if split == 'train':
            ndarray = ndarray[:train_size]
        else:
            ndarray = ndarray[train_size:]

        # list
        ndarray = ndarray.astype(np.int32)
        pair_list = [(image_names[ndarray[idx, 0] - 1],  # 注意，要-1
                      image_names[ndarray[idx, 1] - 1])  # 注意，要-1
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels


class ZapposV4(BaseSiameseDataset):
    def __init__(self,
                 split,
                 category_id,
                 trans_config=None,
                 include_equal=True,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
                 lx_labels_file_name="zappos-labels-real-lexi.mat",
                 ):
        if trans_config is None:
            trans_config = get_default_transforms_config()
        self._split = split
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name
        self._lx_labels_file_name = lx_labels_file_name

        super(ZapposV4, self).__init__(split, category_id, trans_config)

    def _get_splits(self):
        return SPLITS

    def _get_categories(self):
        return CATEGORIES

    def _get_list_and_labels(self, split):
        def _convert_winnter(old):
            # original: left -> 1, right -> 2, equal -> 3
            # target: left -> -1, right -> 1, equal -> 0
            if old == 3:
                return 0
            if old == 1:
                return -1
            return 1

        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        ndarray = sio.loadmat(
            opj(annotation_dir, self._labels_file_name)
        )['mturkOrder'].flatten()[self._category_id]
        if self._include_equal:
            ndarray2 = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]
            attr_dict = {0: 'open', 1: 'pointy', 2: 'sporty', 3: 'comfort'}
            tmp = mat73.loadmat(
                os.path.join(annotation_dir, self._lx_labels_file_name)
            )
            tmp_name = tmp['attrNames']
            for idx, name in enumerate(tmp_name):
                if attr_dict[self._category_id] == name:
                    attr_idx = idx
            ndarray3 = tmp['mturkOrder'][attr_idx]
            ndarray = np.concatenate([ndarray3, ndarray2, ndarray])
        print(ndarray.shape, ndarray3.shape, ndarray[-1], ndarray2[0], ndarray3[0])
        split_ratio = 0.9
        train_size = int(len(ndarray)*split_ratio)
        np.random.shuffle(ndarray)
        if split == 'train':
            ndarray = ndarray[:train_size]
        else:
            ndarray = ndarray[train_size:]

        # list
        ndarray = ndarray.astype(np.int32)
        pair_list = [(image_names[ndarray[idx, 0] - 1],  # 注意，要-1
                      image_names[ndarray[idx, 1] - 1])  # 注意，要-1
                     for idx in range(ndarray.shape[0])]

        # labels
        labels = ndarray[:, 3]
        labels = [_convert_winnter(l) for l in labels]

        return pair_list, labels


class ZapposV3Predict(BasePredictDataset):
    def __init__(self,
                 category_id,
                 include_equal=True,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
                 lx_labels_file_name="zappos-labels-real-lexi.mat",
                 ):
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name
        self._lx_labels_file_name = lx_labels_file_name

        super(ZapposV3Predict, self).__init__(min_height, min_width, is_bgr)

    def _get_categories(self):
        return CATEGORIES

    def _get_image_full_paths(self):
        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        ndarray = sio.loadmat(
            opj(annotation_dir, self._labels_file_name)
        )['mturkOrder'].flatten()[self._category_id]
        if self._include_equal:
            ndarray2 = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]
            ndarray = np.concatenate([ndarray2, ndarray])
        print(ndarray.shape, ndarray[-1], ndarray2[0])
        split_ratio = 0.9
        train_size = int(len(ndarray)*split_ratio)
        np.random.shuffle(ndarray)
        # list
        ndarray = np.asarray(ndarray, np.int32)
        image_list = []
        for idx in range(ndarray.shape[0]):
            image_list.append(image_names[ndarray[idx, 0] - 1])
            image_list.append(image_names[ndarray[idx, 1] - 1])

        return image_list


class ZapposV4Predict(BasePredictDataset):
    def __init__(self,
                 category_id,
                 include_equal=True,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 labels_file_name="zappos-labels.mat",
                 fg_labels_file_name="zappos-labels-fg.mat",
                 lx_labels_file_name="zappos-labels-real-lexi.mat",
                 ):
        self._category_id = category_id
        self._include_equal = include_equal
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        self._labels_file_name = labels_file_name
        self._fg_labels_file_name = fg_labels_file_name
        self._lx_labels_file_name = lx_labels_file_name

        super(ZapposV4Predict, self).__init__(min_height, min_width, is_bgr)

    def _get_categories(self):
        return CATEGORIES

    def _get_image_full_paths(self):
        # dirs
        opj = os.path.join
        image_dir = opj(self._dataset_dir, self._image_dir_name)
        annotation_dir = opj(
            self._dataset_dir, self._annoatation_dir_name)

        # image names list
        image_names_raw = sio.loadmat(
            opj(annotation_dir, self._image_names_file_name)
        )['imagepath'].flatten()
        image_names = _get_image_names(image_names_raw, image_dir)

        # pairs and ndarray
        ndarray = sio.loadmat(
            opj(annotation_dir, self._labels_file_name)
        )['mturkOrder'].flatten()[self._category_id]
        if self._include_equal:
            ndarray2 = sio.loadmat(
                os.path.join(annotation_dir, self._fg_labels_file_name)
            )['mturkHard'].flatten()[self._category_id]
            attr_dict = {0: 'open', 1: 'pointy', 2: 'sporty', 3: 'comfort'}
            tmp = mat73.loadmat(
                os.path.join(annotation_dir, self._lx_labels_file_name)
            )
            tmp_name = tmp['attrNames']
            for idx, name in enumerate(tmp_name):
                if attr_dict[self._category_id] == name:
                    attr_idx = idx
            ndarray3 = tmp['mturkOrder'][attr_idx]
            ndarray = np.concatenate([ndarray3, ndarray2, ndarray])
        print(ndarray.shape, ndarray3.shape, ndarray[-1], ndarray2[0], ndarray3[0])
        split_ratio = 0.9
        train_size = int(len(ndarray)*split_ratio)
        np.random.shuffle(ndarray)
        # list
        ndarray = np.asarray(ndarray, np.int32)
        image_list = []
        for idx in range(ndarray.shape[0]):
            image_list.append(image_names[ndarray[idx, 0] - 1])
            image_list.append(image_names[ndarray[idx, 1] - 1])

        return image_list


class ZapposPredictDataset(BasePredictDataset):
    def __init__(self,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 ):
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        super(ZapposPredictDataset, self).__init__(
            min_height, min_width, is_bgr
        )

    def _get_image_full_paths(self):
        image_names_raw = sio.loadmat(os.path.join(
            self._dataset_dir,
            self._annoatation_dir_name,
            self._image_names_file_name
        ))['imagepath'].flatten()
        image_list = _get_image_names(
            image_names_raw,
            os.path.join(self._dataset_dir, self._image_dir_name)
        )
        image_list = [i for i in image_list[:1000] if os.path.exists(i)]
        return image_list


class ZapposGenDataset(BasePredictDataset):
    def __init__(self,
                 min_height=224,
                 min_width=224,
                 is_bgr=False,
                 dataset_dir=BASE_DATASET,
                 image_dir_name="ut-zap50k-images",
                 annoatation_dir_name="ut-zap50k-data",
                 image_names_file_name="image-path.mat",
                 ):
        self._dataset_dir = dataset_dir
        self._image_dir_name = image_dir_name
        self._annoatation_dir_name = annoatation_dir_name
        self._image_names_file_name = image_names_file_name
        super(ZapposGenDataset, self).__init__(
            min_height, min_width, is_bgr
        )

    def _get_image_full_paths(self):
        #image_names_raw = sio.loadmat(os.path.join(
        #    self._dataset_dir,
        #    self._annoatation_dir_name,
        #    self._image_names_file_name
        #))['imagepath'].flatten()
        #image_list = _get_image_names(
        #    image_names_raw,
        #    os.path.join(self._dataset_dir, self._image_dir_name)
        #)
        #image_list = [i for i in image_list if os.path.exists(i)]
        root_image_path = '/home/yinyao/Data/Projects/AdversarialRanking/RAL_GNN/dataset/utzap/'
        #img_path_list = [f for f in os.listdir(root_image_path) if os.path.isfile(os.path.join(root_image_path, f))] 
        image_list = glob.glob(os.path.join(root_image_path, '*.png'))

        return image_list

