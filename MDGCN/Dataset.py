import numpy as np
import scipy.sparse as sp
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Dataset:
    def __init__(self, filename, ratio):
        self.name = filename
        self.adj = sp.load_npz("datasets/%s/data/adj.npz" % filename)
        self.features = np.load("datasets/%s/data/features.npy" % filename)
        if ('GCN' in FLAGS.model or 'GMNN' in FLAGS.model or 'MRF' in FLAGS.model):
            self.features = (self.features - np.mean(self.features, axis=0, keepdims=True)) / (
                    np.std(self.features, axis=0, keepdims=True) + 1e-10)
            # self.features = self.features / (np.sum(self.features, axis=1, keepdims=True) + 1e-8)
        else:
            self.features = (self.features - np.min(self.features)) / (np.max(self.features) - np.min(self.features))
        self.label = np.load("datasets/%s/data/label.npy" % filename)
        print("num users", len(self.label))
        idx = np.random.permutation(len(self.label))
        self.train_idx = idx[:int(len(self.label) * 0.4 * FLAGS.ratio)]
        self.val_idx = idx[int(len(self.label) * 0.4):int(len(self.label) * 0.5)]
        self.test_idx = idx[int(len(self.label) * 0.5):]
