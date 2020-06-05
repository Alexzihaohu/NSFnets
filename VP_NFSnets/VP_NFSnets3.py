# author Zihao Hu
# time 5/12/2020

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
        self.loss = alpha * tf.reduce_mean(tf.square(self.u_ini_tf - self.u_ini_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.v_ini_tf - self.v_ini_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.w_ini_tf - self.w_ini_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    beta * tf.reduce_mean(tf.square(self.w_boundary_tf - self.w_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_w_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))

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

        Re = 1

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

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1/Re * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1/Re * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_e


# 需要除去 lambda_1

    def callback(self, loss):
        print('Loss: %.3e' % loss)

# train的tf_dict需要修改

    def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0, self.z_ini_tf: self.z0, self.t_ini_tf: self.t0,
                   self.u_ini_tf: self.u0, self.v_ini_tf: self.v0, self.w_ini_tf: self.w0,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t, self.learning_rate: learning_rate}

# tf_dict应该是投喂数据的过程

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

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def BFGS_train(self):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0, self.z_ini_tf: self.z0, self.t_ini_tf: self.t0,
                   self.u_ini_tf: self.u0, self.v_ini_tf: self.v0, self.w_ini_tf: self.w0,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
    #
    # def train(self, epoch=10, nIter=150, learning_rate=1e-3):
    #
    #     for ep in range(epoch):
    #
    #         batch_size1 = len(self.x0) // nIter
    #         batch_size2 = len(self.xb) // nIter
    #         batch_size3 = len(self.x) // nIter
    #
    #         arr1 = np.arange(batch_size1 * nIter)
    #         arr2 = np.arange(batch_size2 * nIter)
    #         arr3 = np.arange(batch_size3 * nIter)
    #
    #         permu1 = np.random.permutation(arr1).reshape((nIter, batch_size1))
    #         permu2 = np.random.permutation(arr2).reshape((nIter, batch_size2))
    #         permu3 = np.random.permutation(arr3).reshape((nIter, batch_size3))
    #
    #         start_time = time.time()
    #         for it in range(nIter):
    #             tf_dict = {self.x_ini_tf: self.x0[permu1[it, :], :],
    #                        self.y_ini_tf: self.y0[permu1[it, :], :],
    #                        self.z_ini_tf: self.z0[permu1[it, :], :],
    #                        self.t_ini_tf: self.t0[permu1[it, :], :],
    #                        self.u_ini_tf: self.u0[permu1[it, :], :],
    #                        self.v_ini_tf: self.v0[permu1[it, :], :],
    #                        self.w_ini_tf: self.w0[permu1[it, :], :],
    #                        self.x_boundary_tf: self.xb[permu2[it, :], :],
    #                        self.y_boundary_tf: self.yb[permu2[it, :], :],
    #                        self.z_boundary_tf: self.zb[permu2[it, :], :],
    #                        self.t_boundary_tf: self.tb[permu2[it, :], :],
    #                        self.u_boundary_tf: self.ub[permu2[it, :], :],
    #                        self.v_boundary_tf: self.vb[permu2[it, :], :],
    #                        self.w_boundary_tf: self.wb[permu2[it, :], :],
    #                        self.x_tf: self.x[permu3[it, :], :],
    #                        self.y_tf: self.y[permu3[it, :], :],
    #                        self.z_tf: self.z[permu3[it, :], :],
    #                        self.t_tf: self.t[permu3[it, :], :],
    #                        self.learning_rate: learning_rate}
    #
    #             self.sess.run(self.train_op_Adam, tf_dict)
    #
    #             # Print
    #             if it % 10 == 0:
    #                 elapsed = time.time() - start_time
    #                 loss_value = self.sess.run(self.loss, tf_dict)
    #                 print('epoch: %d, It: %d, Loss: %.3e, Time: %.2f' %
    #                       (ep, it, loss_value, elapsed))
    #                 start_time = time.time()
    #
    #     self.optimizer.minimize(self.sess,
    #                             feed_dict=tf_dict,
    #                             fetches=[self.loss],
    #                             loss_callback=self.callback)

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
    def data_generate(x, y, z, t):

        a, d = 1, 1
        u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
        v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
        w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
        p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                             2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                             2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                             2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
            -2 * d * d * t)

        return u, v, w, p

    x1 = np.linspace(-1, 1, 31)
    y1 = np.linspace(-1, 1, 31)
    z1 = np.linspace(-1, 1, 31)
    t1 = np.linspace(0, 1, 11)
    b0 = np.array([-1] * 900)
    b1 = np.array([1] * 900)

    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    zt = np.tile(z1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)
    zt1 = np.tile(z1[1:31], 30)

    xr = x1[0:30].repeat(30)
    yr = y1[0:30].repeat(30)
    zr = z1[0:30].repeat(30)
    xr1 = x1[1:31].repeat(30)
    yr1 = y1[1:31].repeat(30)
    zr1 = z1[1:31].repeat(30)

    train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
    train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
    train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
    train1t = np.tile(t1, 5400)

    train1ub, train1vb, train1wb, train1pb = data_generate(train1x, train1y, train1z, train1t)

    xb_train = train1x.reshape(train1x.shape[0], 1)
    yb_train = train1x.reshape(train1y.shape[0], 1)
    zb_train = train1x.reshape(train1z.shape[0], 1)
    tb_train = train1x.reshape(train1t.shape[0], 1)
    ub_train = train1x.reshape(train1ub.shape[0], 1)
    vb_train = train1x.reshape(train1vb.shape[0], 1)
    wb_train = train1x.reshape(train1wb.shape[0], 1)
    pb_train = train1x.reshape(train1pb.shape[0], 1)

    x_0 = np.tile(x1, 31 * 31)
    y_0 = np.tile(y1.repeat(31), 31)
    z_0 = z1.repeat(31 * 31)
    t_0 = np.array([0] * x_0.shape[0])

    u_0, v_0, w_0, p_0 = data_generate(x_0, y_0, z_0, t_0)

    u0_train = u_0.reshape(u_0.shape[0], 1)
    v0_train = v_0.reshape(v_0.shape[0], 1)
    w0_train = w_0.reshape(w_0.shape[0], 1)
    p0_train = p_0.reshape(p_0.shape[0], 1)
    x0_train = x_0.reshape(x_0.shape[0], 1)
    y0_train = y_0.reshape(y_0.shape[0], 1)
    z0_train = z_0.reshape(z_0.shape[0], 1)
    t0_train = t_0.reshape(t_0.shape[0], 1)
    # Rearrange Data

    # unsupervised part

    xx = np.random.randint(31, size=10000) / 15 - 1
    yy = np.random.randint(31, size=10000) / 15 - 1
    zz = np.random.randint(31, size=10000) / 15 - 1
    tt = np.random.randint(11, size=10000) / 10

    uu, vv, ww, pp = data_generate(xx, yy, zz, tt)

    x_train = xx.reshape(xx.shape[0], 1)
    y_train = yy.reshape(yy.shape[0], 1)
    z_train = zz.reshape(zz.shape[0], 1)
    t_train = tt.reshape(tt.shape[0], 1)

    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers)

    model.Adam_train(5000, 1e-3)
    model.Adam_train(5000, 1e-4)
    model.Adam_train(50000, 1e-5)
    model.Adam_train(50000, 1e-6)
    model.BFGS_train()

    # Test Data
    x_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    y_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    z_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    t_star = np.random.randint(11, size=(100, 1)) / 10

    u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)

    # Prediction
    u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star, t_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error v: %e' % error_w)
    print('Error p: %e' % error_p)
