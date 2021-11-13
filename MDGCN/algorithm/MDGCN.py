import os
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import scipy.sparse as sp
import math
from sklearn import metrics

flags = tf.flags
FLAGS = flags.FLAGS


class MDGCN:
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = dataset.features.copy()
        self.num_users = len(self.features)
        self.adj = dataset.adj.toarray()
        self.adj_sp = dataset.adj.astype("float32")
        self.label = (dataset.label).astype(np.int)
        self.label_onehot = np.eye(2)[self.label]
        self.N, self.D = self.features.shape
        self.K = 2
        self.hidden_size = [FLAGS.hiddens]
        self.dropout = FLAGS.dropout
        self.train_idx_np = self.dataset.train_idx
        self.q_features = tf.constant(self.features, tf.float32)
        self.test_idx_np = self.dataset.test_idx
        self.p_hidden_size = FLAGS.hiddens
        self.p_features_np = 0
        self.get_adj()
        self.get_idx()

        self.build_graph()

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.training = tf.placeholder_with_default(False, shape=())
            self.train_idx = tf.placeholder(tf.int32, shape=[None])
            self.p_features = tf.placeholder(tf.float32)
            self.q_label = tf.placeholder(tf.float32)
            self.p_label = tf.placeholder(tf.float32)
            self.q_important = tf.placeholder_with_default(np.ones((self.N, 1), dtype=np.float32), shape=(self.N, 1))

    def create_weight_terms(self, flag, d):
        w_init = slim.xavier_initializer
        previous_size = d
        weights = []
        biases = []
        for idx, layer_size in enumerate(self.hidden_size):
            weight = tf.get_variable(f"W_{flag}_{idx + 1}", shape=[previous_size, layer_size], dtype=tf.float32,
                                     initializer=w_init())
            bias = tf.get_variable(f"b_{flag}_{idx + 1}", shape=[layer_size], dtype=tf.float32,
                                   initializer=w_init())
            weights.append(weight)
            biases.append(bias)
            previous_size = layer_size
        weight_final = tf.get_variable(f"W_{flag}_{len(self.hidden_size) + 1}", shape=[previous_size, self.K],
                                       dtype=tf.float32,
                                       initializer=w_init())
        bias_final = tf.get_variable(f"b_{flag}_{len(self.hidden_size) + 1}", shape=[self.K], dtype=tf.float32,
                                     initializer=w_init())
        weights.append(weight_final)
        biases.append(bias_final)
        return weights, biases

    def create_q_model(self):
        self.q_weights_bi, self.q_biases_bi = self.create_weight_terms("q_bi", self.D)
        self.q_weights_in, self.q_biases_in = self.create_weight_terms("q_in", self.D)
        self.q_weights_out, self.q_biases_out = self.create_weight_terms("q_out", self.D)
        self.q_weights = self.q_weights_bi + self.q_weights_in + self.q_weights_out + self.q_biases_bi + self.q_biases_in + self.q_biases_out
        self.q_features_dropout = tf.nn.dropout(self.q_features, rate=self.dropout)
        self.q_attrs_comp = tf.cond(self.training,
                                    lambda: self.q_features_dropout,
                                    lambda: self.q_features) if self.dropout > 0. else self.q_features
        with tf.variable_scope('q_model'):
            hidden = self.q_attrs_comp
            for ix in range(len(self.hidden_size)):
                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_bi_norm, hidden @ self.q_weights_bi[ix]) + \
                          self.q_biases_bi[ix]
                hidden1 += tf.sparse_tensor_dense_matmul(self.adj_in_norm, hidden @ self.q_weights_in[ix]) + \
                           self.q_biases_in[ix]
                hidden1 += tf.sparse_tensor_dense_matmul(self.adj_out_norm, hidden @ self.q_weights_out[ix]) + \
                           self.q_biases_out[ix]
                hidden = tf.nn.relu(hidden1)
                if (ix == 0):
                    self.q_hiddens = hidden
                    self.q_hiddens1 = hidden1
                hidden_dropout = tf.nn.dropout(hidden, rate=self.dropout)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if self.dropout > 0. else hidden
        self.q_logits = tf.sparse_tensor_dense_matmul(self.adj_bi_norm, hidden @ self.q_weights_bi[-1]) + \
                        self.q_biases_bi[-1]
        self.q_logits += tf.sparse_tensor_dense_matmul(self.adj_in_norm, hidden @ self.q_weights_in[-1]) + \
                         self.q_biases_in[-1]
        self.q_logits += tf.sparse_tensor_dense_matmul(self.adj_out_norm, hidden @ self.q_weights_out[-1]) + \
                         self.q_biases_out[-1]
        self.q_predict = tf.nn.softmax(self.q_logits)
        if (FLAGS.mrf):
            self.q_predict = self.create_mrf(self.q_predict, True)
            self.q_weights += [self.mrf_weight1, self.mrf_weight2, self.mrf_weight3]

    def create_p_model(self):
        self.p_weights_bi, self.p_biases_bi = self.create_weight_terms("p_bi", self.p_hidden_size)
        self.p_weights_in, self.p_biases_in = self.create_weight_terms("p_in", self.p_hidden_size)
        self.p_weights_out, self.p_biases_out = self.create_weight_terms("p_out", self.p_hidden_size)
        self.p_weights = self.p_weights_bi + self.p_weights_in + self.p_weights_out + self.p_biases_bi + self.p_biases_in + self.p_biases_out
        self.p_features_dropout = tf.nn.dropout(self.p_features, rate=self.dropout)
        self.p_attrs_comp = tf.cond(self.training,
                                    lambda: self.p_features_dropout,
                                    lambda: self.p_features) if self.dropout > 0. else self.p_features
        with tf.variable_scope('p_model'):
            hidden = self.p_attrs_comp
            for ix in range(len(self.hidden_size)):
                hidden1 = tf.sparse_tensor_dense_matmul(self.adj_bi_norm, hidden @ self.p_weights_bi[ix]) + \
                          self.p_biases_bi[ix]
                hidden1 += tf.sparse_tensor_dense_matmul(self.adj_in_norm, hidden @ self.p_weights_in[ix]) + \
                           self.p_biases_in[ix]
                hidden1 += tf.sparse_tensor_dense_matmul(self.adj_out_norm, hidden @ self.p_weights_out[ix]) + \
                           self.p_biases_out[ix]
                hidden = tf.nn.relu(hidden1)
                hidden_dropout = tf.nn.dropout(hidden, rate=self.dropout)
                hidden = tf.cond(self.training,
                                 lambda: hidden_dropout,
                                 lambda: hidden) if self.dropout > 0. else hidden
        self.p_logits = tf.sparse_tensor_dense_matmul(self.adj_bi_norm, hidden @ self.p_weights_bi[-1]) + \
                        self.p_biases_bi[-1]
        self.p_logits += tf.sparse_tensor_dense_matmul(self.adj_in_norm, hidden @ self.p_weights_in[-1]) + \
                         self.p_biases_in[-1]
        self.p_logits += tf.sparse_tensor_dense_matmul(self.adj_out_norm, hidden @ self.p_weights_out[-1]) + \
                         self.p_biases_out[-1]
        self.p_predict = tf.nn.softmax(self.p_logits)
        if (FLAGS.mrf):
            self.p_predict = self.create_mrf(self.p_predict)
            self.p_weights += [self.mrf_weight1, self.mrf_weight2, self.mrf_weight3]

    def update_p(self):
        predicts = self.sess.run(self.q_predict)
        self.p_label_np = predicts.copy() > 0.5
        self.p_label_np[self.train_idx_np] = self.label_onehot[self.train_idx_np]
        hiddens = self.sess.run(self.q_hiddens)
        p_features_np = np.zeros((self.N, self.p_hidden_size))
        for i in range(len(hiddens)):
            p_features_np[i] = np.mean(hiddens[np.array(self.idx_list[i])], axis=0)
        self.p_features_np = p_features_np
        self.p_features_np = (self.p_features_np - np.mean(self.p_features_np, axis=0)) / (
                np.std(self.p_features_np, axis=0) + 1e-8)
        np.save("plabel.npy", self.p_label_np)
        np.save("pfeatures.npy", self.p_features_np)

    def update_q(self):
        predicts = self.sess.run(self.p_predict, feed_dict={self.p_features: self.p_features_np})
        self.q_label_np = predicts.copy()
        self.q_label_np[self.train_idx_np] = self.label_onehot[self.train_idx_np]

    def create_mrf_weights(self):
        w_init = slim.xavier_initializer
        self.mrf_weight1 = tf.get_variable(f"W1", shape=[2, 2], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(0., 0.1))
        self.mrf_weight2 = tf.get_variable(f"W2", shape=[2, 2], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(0., 0.1))
        self.mrf_weight3 = tf.get_variable(f"W3", shape=[1], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(0., 0.1))

        self.w_in = self.mrf_weight1 * tf.constant(np.array([[-1, 1], [-1, -1]]), dtype=tf.float32)
        self.w_bi = self.mrf_weight2 * tf.constant(np.array([[-1, 1], [1, -1]]), dtype=tf.float32)
        self.w_out = tf.transpose(self.w_in)

        self.adj_bi = tf.SparseTensor(np.array(self.adj_bi_sp.nonzero()).T,
                                      self.adj_bi_sp.data,
                                      [self.num_users, self.num_users])
        self.adj_in = tf.SparseTensor(np.array(self.adj_in_sp.nonzero()).T,
                                      self.adj_in_sp.data,
                                      [self.num_users, self.num_users])
        self.adj_out = tf.SparseTensor(np.array(self.adj_out_sp.nonzero()).T,
                                       self.adj_out_sp.data,
                                       [self.num_users, self.num_users])

    def create_mrf(self, H, flag=False):
        iters = 5
        Q = H
        for i in range(iters):
            Q1 = tf.log(H + 1e-8)
            Q2 = -tf.sparse_tensor_dense_matmul(self.adj_in, Q @ self.w_in) - tf.sparse_tensor_dense_matmul(
                self.adj_out, Q @ self.w_out) - tf.sparse_tensor_dense_matmul(self.adj_bi, Q @ self.w_bi)
            Q = tf.nn.softmax(
                Q1 * tf.sigmoid(self.mrf_weight3) + tf.identity(Q2) * (1. - tf.sigmoid(self.mrf_weight3)))
            if (flag == True):
                self.q_logits = Q1 * tf.sigmoid(self.mrf_weight3) + tf.identity(Q2) * (
                        1. - tf.sigmoid(self.mrf_weight3))
        return Q

    def create_optimizer(self):
        lr = 0.01
        if (FLAGS.dataset == '1KS-10KN'):
            lr = 0.002
        with tf.variable_scope('loss'):
            self.q_label_gather = tf.gather(self.q_label, self.train_idx)
            self.q_predict_gather = tf.gather(self.q_predict, self.train_idx)
            self.q_important_gather = tf.gather(self.q_important, self.train_idx)
            self.q_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.q_weights]) * 5e-4
            self.q_loss += tf.reduce_mean(
                -self.q_label_gather * tf.log(self.q_predict_gather + 1e-8) * self.q_important_gather)
            self.q_optimizer = tf.train.AdamOptimizer(lr)
            self.q_train_op = self.q_optimizer.minimize(self.q_loss, name='q_optimizer', var_list=self.q_weights)

            self.p_label_gather = tf.gather(self.p_label, self.train_idx)
            self.p_predict_gather = tf.gather(self.p_predict, self.train_idx)
            self.p_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.p_weights]) * 5e-4
            self.p_loss += tf.reduce_mean(-self.p_label_gather * tf.log(self.p_predict_gather + 1e-8))
            self.p_optimizer = tf.train.AdamOptimizer(lr)
            self.p_train_op = self.p_optimizer.minimize(self.p_loss, name='p_optimizer', var_list=self.p_weights)

    def build_graph(self):
        self.create_placeholders()
        self.create_mrf_weights()
        self.create_q_model()
        self.create_p_model()
        self.create_optimizer()

    def train(self, nb_epochs=150):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        pre_train = 2400
        if (FLAGS.gmnn):
            pre_train = 400
        bestauc = -1
        bestacc = -1
        bestrecall = -1
        auc_list = []
        # pre-training
        for cur_epochs in range(pre_train):
            feed_train = {self.train_idx: self.train_idx_np, self.q_label: self.label_onehot,
                          self.training: True}
            self.sess.run([self.q_train_op], feed_dict=feed_train)
            acc, auc, recall = self.evaluate(self.sess.run(self.q_predict))
            if (acc >= bestacc):
                bestacc = acc
            if (auc > bestauc):
                bestauc = auc
            if (recall > bestrecall):
                bestrecall = recall
            if (cur_epochs % 20 == 0):
                print("pre-train epochs:", cur_epochs)
                print(acc, auc, recall, bestacc, bestauc, bestrecall)
            auc_list.append(recall)

        if (FLAGS.gmnn):
            bestauc = -1
            bestacc = -1
            bestrecall = -1
            if (FLAGS.dataset == 'TwitterSH'):
                EM_epochs = 40
            else:
                EM_epochs = 10
            # EM_epochs=2000//FLAGS.epochs
            for iters in range(EM_epochs):
                # M step
                self.update_p()
                self.cur_best = -1
                for cur_epochs in range(2000 // EM_epochs + (1 - np.sign(iters)) * 400):
                    train_idx = np.arange(self.N)
                    feed_train = {self.train_idx: train_idx, self.p_label: self.p_label_np,
                                  self.training: True, self.p_features: self.p_features_np}
                    _, = self.sess.run([self.p_train_op], feed_dict=feed_train)
                    acc, auc, recall = self.evaluate(
                        self.sess.run(self.p_predict, feed_dict={self.p_features: self.p_features_np}))
                    if (acc >= bestacc):
                        bestacc = acc
                    if (auc > bestauc):
                        bestauc = auc
                    if (recall > bestrecall):
                        bestrecall = recall
                    if (cur_epochs % 20 == 0):
                        print("p model epochs:", iters, cur_epochs)
                        print(acc, auc, recall, bestacc, bestauc, bestrecall)

                # E step
                self.update_q()
                idx1 = np.where(self.q_label_np[:, 1] >= 0.5)[0]
                idx0 = np.where(self.q_label_np[:, 1] < 0.5)[0]
                train_idx1 = np.random.choice(idx1, len(idx0))
                train_idx = np.concatenate([idx0, train_idx1])
                for cur_epochs in range(2000 // EM_epochs):
                    feed_train = {self.train_idx: train_idx, self.q_label: self.q_label_np,
                                  self.training: True}
                    self.sess.run([self.q_train_op], feed_dict=feed_train)
                    acc, auc, recall = self.evaluate(self.sess.run(self.q_predict))
                    if (acc >= bestacc):
                        bestacc = acc
                    if (auc > bestauc):
                        bestauc = auc
                    if (recall > bestrecall):
                        bestrecall = recall
                    if (cur_epochs % 20 == 0):
                        print("q model epochs:", iters, cur_epochs)
                        print(acc, auc, recall, bestacc, bestauc, bestrecall)
                    auc_list.append(recall)
        np.save("results/f1_%s_%d_%d_%.2f.npy" % (FLAGS.dataset, FLAGS.mrf, FLAGS.gmnn, FLAGS.ratio),
                np.array(auc_list))
        features = self.sess.run(self.q_hiddens1)
        np.save("features/%s_%d_%d_%.2f.npy" % (FLAGS.dataset, FLAGS.mrf, FLAGS.gmnn, FLAGS.ratio), features)
        return bestacc, bestauc, bestrecall

    def evaluate(self, predict):
        test_idx = self.dataset.test_idx
        test_auc = metrics.average_precision_score(self.label[test_idx], predict[test_idx, 1])
        test_recall = metrics.f1_score(self.label[test_idx], predict[test_idx, 1] >= 0.5)
        test_acc = metrics.accuracy_score(self.label[test_idx], predict[test_idx, 1] >= 0.5)
        return test_acc, test_auc, test_recall

    def get_adj(self):
        adj_bi = np.zeros_like(self.adj, dtype=np.float)
        adj_in = np.zeros_like(self.adj, dtype=np.float)
        adj_out = np.zeros_like(self.adj, dtype=np.float)

        coo = self.dataset.adj.tocoo()
        rows = coo.row
        cols = coo.col
        for i in range(len(rows)):
            if (self.adj[cols[i], rows[i]] == 1):
                adj_bi[rows[i], cols[i]] = 1.
                adj_bi[cols[i], rows[i]] = 1.
            else:
                adj_out[rows[i], cols[i]] = 1.
                adj_in[cols[i], rows[i]] = 1.
        self.adj_bi_sp = sp.csr_matrix(adj_bi).astype("float32")
        self.adj_in_sp = sp.csr_matrix(adj_in).astype("float32")
        self.adj_out_sp = sp.csr_matrix(adj_out).astype("float32")

        adj_ = self.adj_bi_sp + sp.eye(self.adj_bi_sp.shape[0])
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        self.adj_bi_norm = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr().astype("float32")

        adj_ = self.adj_in_sp
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.))
        self.adj_in_norm = degree_mat_inv_sqrt.dot(adj_).tocsr().astype("float32")

        adj_ = self.adj_out_sp
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.))
        self.adj_out_norm = degree_mat_inv_sqrt.dot(adj_).tocsr().astype("float32")

        adj_ = self.adj_sp
        rowsum = adj_.sum(1).A1 + 1e-8
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.))
        self.adj_norm = degree_mat_inv_sqrt.dot(adj_).tocsr().astype("float32")

        self.adj_bi_norm = tf.SparseTensor(np.array(self.adj_bi_norm.nonzero()).T,
                                           self.adj_bi_norm[self.adj_bi_norm.nonzero()].A1,
                                           [self.N, self.N])
        self.adj_in_norm = tf.SparseTensor(np.array(self.adj_in_norm.nonzero()).T,
                                           self.adj_in_norm[self.adj_in_norm.nonzero()].A1,
                                           [self.N, self.N])
        self.adj_out_norm = tf.SparseTensor(np.array(self.adj_out_norm.nonzero()).T,
                                            self.adj_out_norm[self.adj_out_norm.nonzero()].A1,
                                            [self.N, self.N])
        self.adj_norm = tf.SparseTensor(np.array(self.adj_norm.nonzero()).T,
                                        self.adj_norm[self.adj_norm.nonzero()].A1,
                                        [self.N, self.N])

    def get_P(self):
        P_in = np.zeros((2, 2))
        adj_in = self.adj_in_sp.toarray()
        for i in range(self.N):
            idx = np.where(adj_in[i] != 0)[0]
            P_in[0, self.label[i]] += len(idx) - np.sum(self.label[idx])
            P_in[1, self.label[i]] += np.sum(self.label[idx])
        P_in = P_in / np.sum(P_in, axis=0, keepdims=True)
        P_out = P_in.T

        P_bi = np.zeros((2, 2))
        adj_bi = self.adj_bi_sp.toarray()
        for i in range(self.N):
            idx = np.where(adj_bi[i] != 0)[0]
            P_bi[0, self.label[i]] += len(idx) - np.sum(self.label[idx])
            P_bi[1, self.label[i]] += np.sum(self.label[idx])
        P_bi = P_bi / np.sum(P_bi, axis=0, keepdims=True)
        return P_in, P_out, P_bi

    def get_idx(self):
        adj = self.adj + self.adj.T
        idx_list = []
        for i in range(self.N):
            idx = []
            idx += list(np.where(adj[i] != 0)[0])
            idx.append(i)
            idx_list.append(idx)
        self.idx_list = idx_list

    def get_similar(self, indices):
        rows = indices[:, 0]
        cols = indices[:, 1]

        feature1 = tf.gather(self.q_hiddens, rows)
        feature2 = tf.gather(self.q_hiddens, cols)
        similar = tf.reduce_sum(feature1 * feature2, axis=1) / tf.sqrt(
            tf.reduce_sum(tf.square(feature1), axis=1) + 1e-8) / tf.sqrt(
            tf.reduce_sum(tf.square(feature2), axis=1) + 1e-8)
        similar = tf.stop_gradient(similar)
        return similar
