import tensorflow as tf
from utils import *
import os
from model import ESPCN
import numpy as np
import matplotlib.pyplot as plt
def loaddata():
    data_dir = os.path.join(os.getcwd(),"2.bmp")
    sequence = make_data(data_dir)
    arrinput = np.asarray(sequence)
    print(arrinput.shape)
    make_hf(arrinput)

def make_data(data):
    sub_input_sequence = []
    input_ = preprocess(data)  # do bicbuic
    input_ = input_ / 255.0
    sub_input_sequence.append(input_)
    return sub_input_sequence
def preprocess(data):
    data=cv2.imread(data)
    input = cv2.resize(data,(200,200),interpolation=cv2.INTER_AREA)

    cv2.imwrite("./input.bmp",input)
    return input

def make_hf(input_):
    savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    cv2.imwrite(os.path.join(os.getcwd(), path), image * 255.)
    # NOTE: because normial, we need mutlify 255 back
def read_data():
    path = os.path.join(os.getcwd(),'checkpoint\\test.h5')
    print(path)
    with h5py.File(path,'r') as hf:
        input = np.array(hf.get('input'))
        return input
def _phase_shift_test(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (1, a, b, r, r))
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
    X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
    return tf.reshape(X, (1, a * r, b * r, 1))
def PS( X, r):
    # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, 3, 3)#把最后一个卷积出来的图片进行分割成3个张量在3这个维度,[None,height,weight,out_channel]X:,就是把特征图片三等分，这就是意味着我们可以扩大三倍，哈哈是不是跟scale对应了，w为[None,17,17,3]
    X = tf.concat([_phase_shift_test(x, r) for x in Xc], 3)  # Do the concat RGB
    return X



if __name__ == '__main__':
    input = read_data()
    model_path = os.path.join(os.getcwd(),"checkpoint\espcn_17_3")
    print(model_path)


    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./checkpoint/espcn_17_3/ESPCN.model-27000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/espcn_17_3'))
        graph = tf.get_default_graph()
        images = tf.placeholder(tf.float32, [None, 200, 200, 3], name='images')
        weights = {
            'w1': graph.get_tensor_by_name("w1:0"),
            'w2': graph.get_tensor_by_name("w2:0"),
            'w3': graph.get_tensor_by_name("w3:0")
        }
        biases = {
            'b1': graph.get_tensor_by_name("b1:0"),
            'b2': graph.get_tensor_by_name("b2:0"),
            'b3': graph.get_tensor_by_name("b3:0")
        }

        conv1 = tf.nn.relu(tf.nn.conv2d(images, weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b2'])
        conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b3']
        ps = PS(conv3, 3)
        a = tf.nn.tanh(ps)
        b = sess.run(a,feed_dict = {images:input})
        b = np.squeeze(b)
        b = b*255.
        plt.imshow(b)
        plt.show()
