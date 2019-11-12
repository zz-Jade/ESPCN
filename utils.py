import cv2
import numpy as np
import tensorflow as tf
import os
import glob
import h5py
import config



def input_setup():
    data = prepare_data()
    print("1")
    sub_input_sequence,sub_label_sequence = make_sub_data(data)
    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence)  # [?, 17, 17, 3]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 17 * scale , 17 * scale, 3]
    print(arrinput.shape)
    make_data_hf(arrinput, arrlabel)


def prepare_data():
    data_dir = os.path.join(os.getcwd(),config.dataset)
    data = glob.glob(os.path.join(data_dir,"*.bmp"))
    print(len(data))
    return data

def imread(path):
    img = cv2.imread(path)
    return img

def modcrop(img,scale = 3):
    h, w, _ = img.shape
    h = int(h / scale) * scale
    w = int(w / scale) * scale
    img = img[0:h, 0:w, :]
    return  img

def preprocess(path,scale = 3):
    img = imread(path)
    label_ = modcrop(img,scale)
    input_ = cv2.resize(label_,None,fx = 1.0/scale,fy = 1.0/scale, interpolation=cv2.INTER_CUBIC)
    return input_,label_

def make_sub_data(data):
    sub_input_sequence = []
    sub_label_sequence = []
    for i in range(len(data)):
        j = 0
        input_, label_, = preprocess(data[i], config.scale)# do bicbuic
        # cv2.imwrite("./preprocess/input{}.png".format(i),input_)
        # cv2.imwrite("./preprocess/label{}.png".format(i), label_)
        if len(input_.shape) == 3:  # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape  # is grayscale
        #把图片全部切成17*17并存取到list中
        #做input的
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):

                sub_input = input_[x: x + config.image_size, y: y + config.image_size,:]
                #cv2.imwrite("./sequence/input{}.png".format(j), sub_input_picture)
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_input = sub_input / 255.0
                sub_input_sequence.append(sub_input)
        #做label的所以要大一点，大小根据网络最后的图像大小计算
        for x in range(0, (h * config.scale - config.image_size * config.scale + 1), config.stride * config.scale):
            for y in range(0, (w * config.scale - config.image_size * config.scale + 1), config.stride * config.scale):

                sub_label = label_[x: (x + config.image_size * config.scale),y: (y + config.image_size * config.scale)]  # 17r * 17r
                # Reshape the subinput and sublabel
                sub_label = sub_label.reshape([config.image_size * config.scale, config.image_size * config.scale, config.c_dim])
                # Normialize
                sub_label = sub_label / 255.0
                # Add to sequence
                sub_label_picture = np.array(sub_label)
                sub_label_sequence.append(sub_label)
                j = j+1
                print(j)
    return sub_input_sequence,sub_label_sequence

def make_data_hf(input_,label_):
    savepath = os.path.join(os.getcwd(),'checkpoint\\train.h5')
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)


def read_data():
    path = os.path.join(os.getcwd(),'checkpoint\\train.h5')
    with h5py.File(path,'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_