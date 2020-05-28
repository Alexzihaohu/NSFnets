# author Zihao Hu
# time 5/27/2020

import sys
import tensorflow as tf
import numpy as np
import time

# set random seed

np.random.seed(1234)

tf.set_random_seed(1234)

#############################################
###################VP NSFnet#################
#############################################

class VPNSFnet:
    # Initialize the class
    def __init__(self, x0, y0, z0, t0, u0, v0, w0, xb, yb, zb, tb, ub, vb, wb, x, y, z, t, layers):
        X0 = np.concatenate([x0, y0, z0, t0], 1)  # remove the second bracket
        Xb = np.concatenate([xb, yb, zb, tb], 1)
        X = np.concatenate([x, y, z, t], 1)

        self.lowb = Xb.min(0)  # minimal number in each column
        self.upb = Xb.max(0)

        self.X0 = X0
        self.Xb = Xb
        self.X = X

        self.x0 = X0[:, 0:1]
        self.y0 = X0[:, 1:2]
        self.z0 = X0[:, 2:3]
        self.t0 = X0[:, 3:4]

        self.xb = Xb[:, 0:1]
        self.yb = Xb[:, 1:2]
        self.zb = Xb[:, 2:3]
        self.tb = Xb[:, 3:4]

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.z = X[:, 2:3]
        self.t = X[:, 3:4]

        self.u0 = u0
        self.v0 = v0
        self.w0 = w0

        self.ub = ub
        self.vb = vb
        self.wb = wb

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_ini_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y_ini_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.z_ini_tf = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t_ini_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u_ini_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v_ini_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        self.w_ini_tf = tf.placeholder(tf.float32, shape=[None, self.w0.shape[1]])

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.z_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.zb.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])
        self.w_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.wb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_ini_pred, self.v_ini_pred, self.w_ini_pred, self.p_ini_pred = \
            self.net_NS(self.x_ini_tf, self.y_ini_tf, self.z_ini_tf, self.t_ini_tf)
        self.u_boundary_pred, self.v_boundary_pred, self.w_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf, self.z_boundary_tf, self.t_boundary_tf)
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        alpha = 100
        beta = 100

        # set loss function
        self.loss = alpha * tf.reduce_sum(tf.square(self.u_ini_tf - self.u_ini_pred)) + \
                    alpha * tf.reduce_sum(tf.square(self.v_ini_tf - self.v_ini_pred)) + \
                    alpha * tf.reduce_sum(tf.square(self.w_ini_tf - self.w_ini_pred)) + \
                    beta * tf.reduce_sum(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    beta * tf.reduce_sum(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    beta * tf.reduce_sum(tf.square(self.w_boundary_tf - self.w_boundary_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_w_pred)) + \
                    tf.reduce_sum(tf.square(self.f_e_pred))

        # set optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

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
    # supervised train
    def net_NS(self, x, y, z, t):

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        return u, v, w, p

    # unsupervised train
    def net_f_NS(self, x, y, z, t):

        Re = 999.35

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1 / Re * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1 / Re * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1 / Re * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_e

    # 需要除去 lambda_1

    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    def train(self, epoch=1000, nIter=150, learning_rate=1e-3):
        for ep in range(epoch):
            batch_size1 = len(self.x0) // nIter
            batch_size2 = len(self.xb) // nIter
            batch_size3 = len(self.x) // nIter

            arr1 = np.arange(batch_size1 * nIter)
            arr2 = np.arange(batch_size2 * nIter)
            arr3 = np.arange(batch_size3 * nIter)

            permu1 = np.random.permutation(arr1).reshape((nIter, batch_size1))
            permu2 = np.random.permutation(arr2).reshape((nIter, batch_size2))
            permu3 = np.random.permutation(arr3).reshape((nIter, batch_size3))

            start_time = time.time()
            for it in range(nIter):
                tf_dict = {self.x_ini_tf: self.x0[permu1[it, :], :],
                           self.y_ini_tf: self.y0[permu1[it, :], :],
                           self.z_ini_tf: self.z0[permu1[it, :], :],
                           self.t_ini_tf: self.t0[permu1[it, :], :],
                           self.u_ini_tf: self.u0[permu1[it, :], :],
                           self.v_ini_tf: self.v0[permu1[it, :], :],
                           self.w_ini_tf: self.w0[permu1[it, :], :],
                           self.x_boundary_tf: self.xb[permu2[it, :], :],
                           self.y_boundary_tf: self.yb[permu2[it, :], :],
                           self.z_boundary_tf: self.zb[permu2[it, :], :],
                           self.t_boundary_tf: self.tb[permu2[it, :], :],
                           self.u_boundary_tf: self.ub[permu2[it, :], :],
                           self.v_boundary_tf: self.vb[permu2[it, :], :],
                           self.w_boundary_tf: self.wb[permu2[it, :], :],
                           self.x_tf: self.x[permu3[it, :], :],
                           self.y_tf: self.y[permu3[it, :], :],
                           self.z_tf: self.z[permu3[it, :], :],
                           self.t_tf: self.t[permu3[it, :], :],
                           self.learning_rate: learning_rate}

                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('epoch: %d, It: %d, Loss: %.3e, Time: %.2f' %
                          (ep, it, loss_value, elapsed))
                    start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    # 不需要改变 可能需要注意x_tf等

    def predict(self, x_star, y_star, z_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, w_star, p_star


if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised

    N_train = 10000

    layers = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]

    # Load Data
    train_ini1 = np.load('train_ini1.npy')
    train_iniv1 = np.load('train_iniv1.npy')
    train_inip1 = np.load('train_inip1.npy')
    train_xb1 = np.load('train_xb1.npy')
    train_vb1 = np.load('train_vb1.npy')
    train_pb1 = np.load('train_pb1.npy')

    x0_train = train_ini1[:, 0:1]
    y0_train = train_ini1[:, 1:2]
    z0_train = train_ini1[:, 2:3]
    t0_train = np.zeros(train_ini1[:, 0:1].shape, np.float32)
    u0_train = train_iniv1[:, 0:1]
    v0_train = train_iniv1[:, 1:2]
    w0_train = train_iniv1[:, 2:3]

    xb_train = train_xb1[:, 0:1]
    yb_train = train_xb1[:, 1:2]
    zb_train = train_xb1[:, 2:3]
    tb_train = train_xb1[:, 3:4]
    ub_train = train_vb1[:, 0:1]
    vb_train = train_vb1[:, 1:2]
    wb_train = train_vb1[:, 2:3]

    xnode = np.linspace(12.47, 12.66, 191)
    ynode = np.linspace(-0.9, -0.7, 201)
    znode = np.linspace(4.61, 4.82, 211)

    total_times = np.array(list(range(4000)), dtype=np.float32) * 0.0065

    x_train1 = xnode.reshape(-1, 1)[np.random.choice(191, 20000, replace=True), :]
    y_train1 = ynode.reshape(-1, 1)[np.random.choice(201, 20000, replace=True), :]
    z_train1 = znode.reshape(-1, 1)[np.random.choice(211, 20000, replace=True), :]
    x_train = np.tile(x_train1, (129, 1))
    y_train = np.tile(y_train1, (129, 1))
    z_train = np.tile(z_train1, (129, 1))

    total_times1 = np.array(list(range(129))) * 0.0065
    t_train1 = total_times1.repeat(20000)
    t_train = t_train1.reshape(-1, 1)

    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers)

    model.train(1000, 150, 1e-3)
    model.train(4000, 150, 1e-4)
    model.train(1000, 150, 1e-5)
    model.train(500, 150, 1e-6)

    # # Test Data
    # x_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # y_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # z_star = (np.random.rand(100, 1) - 1 / 2) * 2
    # t_star = np.random.randint(11, size=(100, 1)) / 10
    #
    # u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)
    #
    # # Prediction
    # u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star, t_star)
    #
    # # Error
    # error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    # error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    # error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    # error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    #
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error v: %e' % (error_w))
    # print('Error p: %e' % (error_p))
