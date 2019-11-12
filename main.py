import tensorflow as tf
from model import ESPCN
import config


if __name__ == '__main__':
    with tf.Session() as sess:
        espcn = ESPCN(sess,
                      image_size = config.image_size,
                      scale = config.scale,
                      c_dim = config.c_dim,
                      batch_size = config.batch_size,
                      )
        espcn.train()
