from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random
from Dataset import Dataset
from algorithm.MDGCN import MDGCN

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("gpu", "5", "the gpu id")
flags.DEFINE_string("model", "MDGCN", "the detection model")
flags.DEFINE_string("dataset", "TwitterSH", "the used dataset")
flags.DEFINE_float("ratio", 1., "the ratio of training set")
flags.DEFINE_float("dropout", 0.5, "the ratio of training set")
flags.DEFINE_bool("mrf", False, "use mrf or not")
flags.DEFINE_bool("gmnn", False, "use gmnn or not")
flags.DEFINE_integer("hiddens", 64, "hidden size")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
os.environ['PYTHONHASHSEED'] = str("100")
random.seed(100)
tf.random.set_random_seed(100)
np.random.seed(100)

models = {"MDGCN": MDGCN, "MDGCN-CRF": MDGCN, "ARMGCN": MDGCN, "GCN": MDGCN}
if (FLAGS.model == 'MDGCN'):
    FLAGS.mrf = True
    FLAGS.gmnn = True
if (FLAGS.model == 'MDGCN-CRF'):
    FLAGS.mrf = False
    FLAGS.gmnn = True
if (FLAGS.model == 'GCN'):
    FLAGS.mrf = False
    FLAGS.gmnn = False
if (FLAGS.model == 'ARMGCN'):
    FLAGS.mrf = True
    FLAGS.gmnn = False
dataset = Dataset(FLAGS.dataset, FLAGS.ratio)
acc_list = []
auc_list = []
recall_list = []
for iters in range(30):
    model = models[FLAGS.model](dataset)
    acc, auc, recall = model.train()
    acc_list.append(acc)
    auc_list.append(auc)
    recall_list.append(recall)
    tf.reset_default_graph()
print(np.mean(acc_list), np.mean(auc_list), np.mean(recall_list))
