import tensorflow as tf
import numpy as np
import time
import os
import config
from utils import *



class ESPCN(object):

    def __init__(self,
                 sess,
                 image_size,
                 scale,
                 batch_size,
                 c_dim,
                 test=False
                 ):
        self.sess = sess
        self.image_size = image_size
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.test = test
        self.build_model()
    def build_model(self):
        if not self.test:
            self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32,
                                         [None, self.image_size * self.scale, self.image_size * self.scale, self.c_dim],
                                         name='labels')
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim, 64], stddev=np.sqrt(2.0 / 25 / 3)), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0 / 9 / 64)), name='w2'),
            'w3': tf.Variable(
                tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale], stddev=np.sqrt(2.0 / 9 / 32)),
                name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale], name='b3'))
        }

        self.pred = self.model()

        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver()  # To save checkpoint

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        #r：3
        bsize, a, b, c = I.get_shape().as_list()#返回元组然后变成list，a为高，b为宽，c为特征子图数量
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a * r, b * r, 1))

        # NOTE:test without batchsize

    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)#把最后一个卷积出来的图片进行分割成3个张量在3这个维度,[None,height,weight,out_channel]X:,就是把特征图片三等分，这就是意味着我们可以扩大三倍，哈哈是不是跟scale对应了，w为[None,17,17,3]
        X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)  # Do the concat RGB
        return X
    def model(self):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b1'],name="conv1"),
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b2'],name = "conv2")
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b3']  # This layer don't need ReLU [None,17,17,27]
        ps = self.PS(conv3, self.scale)
        return tf.nn.tanh(ps)
    def train(self):
        input_setup()
        input_,label_ =read_data()
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        print("Training......")
        for ep in range(config.epoch):
            batch_idxs = len(input_) // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images = input_[idx * config.batch_size: (idx + 1) * config.batch_size]
                batch_labels = label_[idx * config.batch_size: (idx + 1) * config.batch_size]
                counter += 1
                _, err = self.sess.run([self.train_op, self.loss],
                                       feed_dict={self.images: batch_images, self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % (
                    (ep + 1), counter, time.time() - time_, err))
                    # print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                if counter % 500 == 0:
                    self.save(config.checkpoint_dir, counter)


    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        checkpoint_dir = os.path.join(os.getcwd(),checkpoint_dir)
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s_%s" % ("espcn", self.image_size, self.scale)  # give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        # Check the checkpoint is exist
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)  # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n" % ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")

    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "ESPCN.model"
        model_dir = "%s_%s_%s" % ("espcn", self.image_size, self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)