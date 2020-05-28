# author Zihao Hu
# time 5/11/2020

import sys
sys.path.append('PINNs-master/Utilities')

import tensorflow as tf
import numpy as np
import time

# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)

class VPNSFnet:
    # Initialize the class
    def __init__(self, xb, yb, ub, vb, x, y, layers):
        # remove the second bracket
        Xb = np.concatenate([xb, yb], 1)
        X = np.concatenate([x, y], 1)

        self.lowb = Xb.min(0)  # minimal number in each column
        self.upb = Xb.max(0)

        self.Xb = Xb
        self.X = X

        self.xb = Xb[:, 0:1]
        self.yb = Xb[:, 1:2]
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]

        self.ub = ub
        self.vb = vb

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # Initialize parameters
        # when applying dynamic weighting
        # self.alpha = tf.Variable([0.0], dtype=tf.float32)
        # self.beta = tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                               log_device_placement=True))

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])

        self.u_boundary_pred, self.v_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf)
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf)

        alpha = 1

        # set loss function

        self.loss = alpha * tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))

        # set optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)# add learning rate here
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)


# do not need adaptation
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

# do not need adaptation

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# do not need adaptation
    def neural_net(self, X, weights, biases):

        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

# ###################without assume###############

    # supervised data-driven
    def net_NS(self, x, y):

        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u, v, p

    # unsupervised train
    def net_f_NS(self, x, y):

        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = (u * u_x + v * u_y) + p_x - (1.0/40) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y - (1.0/40) * (v_xx + v_yy)
        f_e = u_x + v_y

        return u, v, p, f_u, f_v, f_e

    def callback(self, loss):
        print('Loss: %.3e' % loss)

# train

    def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y, self.learning_rate: learning_rate}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

    def BFGS_train(self):

        tf_dict = {self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
# 不需要改变 可能需要注意x_tf等

    def predict(self, x_star, y_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star

if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised
    N_train = 2601
    layers = [2, 50, 50, 50, 50, 3]

    # Load Data
    Re = 40
    lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 1.5, 101)

    yb1 = np.array([-0.5] * 100)
    yb2 = np.array([1] * 100)
    xb1 = np.array([-0.5] * 100)
    xb2 = np.array([1.5] * 100)

    y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
    x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)

    xb_train = x_train1.reshape(x_train1.shape[0], 1)
    yb_train = y_train1.reshape(y_train1.shape[0], 1)
    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)

    x_train = (np.random.rand(N_train, 1) - 1 / 3) * 3 / 2
    y_train = (np.random.rand(N_train, 1) - 1 / 4) * 2

    model = VPNSFnet(xb_train, yb_train, ub_train, vb_train,
                     x_train, y_train, layers)

    model.Adam_train(5000, 1e-3)
    model.Adam_train(5000, 1e-4)
    model.Adam_train(50000, 1e-5)
    model.Adam_train(50000, 1e-6)
    model.BFGS_train()

    # Test Data
    np.random.seed(1234)

    x_star = (np.random.rand(1000, 1) - 1 / 3) * 3 / 2
    y_star = (np.random.rand(1000, 1) - 1 / 4) * 2

    u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
    v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
    p_star = 0.5 * (1 - np.exp(2 * lam * x_star))

    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error p: %e' % error_p)
