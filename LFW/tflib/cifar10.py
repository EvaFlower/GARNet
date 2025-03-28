import numpy as np

import os
import urllib
import gzip
import pickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print('cifar size', images.shape, labels.shape)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) // batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )

def load_data(data_dir):
    train_filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_filename = ['test_batch']
    all_data = []
    all_labels = []
    for filename in train_filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
    train_images = np.concatenate(all_data, axis=0)
    train_labels = np.concatenate(all_labels, axis=0)
    
    all_data = []
    all_labels = []
    for filename in test_filename:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
    test_images = np.concatenate(all_data, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)
    return train_images, train_labels, test_images, test_labels

