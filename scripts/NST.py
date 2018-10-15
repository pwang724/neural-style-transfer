import tensorflow as tf
import numpy as np
from PIL import Image
import PIL
import model

import matplotlib.pyplot as plt
import os, sys
import argparse
import copy

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img', type=str, default='test.jpg', help='content image path')
    parser.add_argument('--style_img', type=str, default='style.jpg', help='style image path')
    parser.add_argument('--model_path', type=str, default="vgg19.npy", help='vgg19 path')
    parser.add_argument('--output', type=str, default='out', help='output folder path')
    parser.add_argument('--lr_rate', type=float, default=1, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=3000, help='Epoch number')
    parser.add_argument('--style_weight', type=float, default=100000, help='trade-off between content and style')
    parser.add_argument('--content_weight', type=float, default=1, help='trade-off between content and style')
    return parser


def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.pause(.1)


# Loss
def gram_matrix(maps):
    if isinstance(maps, tf.Tensor):
        maps_vec = tf.transpose(maps, perm=(0, 3, 1, 2))
        a, b, c, d = maps_vec.shape
        maps_vec = tf.reshape(maps_vec, (a * b, c * d))
        return 1 / (2 * int(a * b * c * d)) * tf.matmul(maps_vec, maps_vec, transpose_b=True)
    else:
        maps_vec = np.array(maps).transpose((0, 3, 1, 2))
        a, b, c, d = maps_vec.shape
        maps_vec = maps_vec.reshape(a * b, c * d)
        return 1 / (2 * (a * b * c * d)) * np.matmul(maps_vec, maps_vec.T)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    dsamp = 2
    dim_img = Image.open(args.content_img)
    img_width, img_height = dim_img.size
    img_width //= dsamp
    img_height //= dsamp

    content_img = Image.open(args.content_img).resize((img_width, img_height), resample=PIL.Image.LANCZOS)
    style_img = Image.open(args.style_img).resize((img_width, img_height), resample=PIL.Image.LANCZOS)
    input_img = copy.copy(content_img)

    # show_image(content_img)
    # show_image(style_img)

    vgg_input = tf.Variable(initial_value=np.zeros(shape=[1, img_height, img_width, 3], dtype='float32'), name='image')
    conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2 = model.build_part_vgg19(vgg_input, params_dir='vgg19.npy')

    # reshape to NHWC
    content_img = np.reshape(content_img, newshape=(-1, img_height, img_width, 3))
    style_img = np.reshape(style_img, newshape=(-1, img_height, img_width, 3))
    input_img = np.reshape(input_img, newshape=(-1, img_height, img_width, 3))
    # input_img = np.random.randint(0, 255, input_img.shape)

    # GPU Config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())

    # Get content feature maps
    sess.run(vgg_input.assign(content_img))
    content_maps_out, = sess.run([conv4_2])

    # Get style feature maps
    sess.run(vgg_input.assign(style_img))
    style_maps_out = sess.run([conv1_1, conv2_1, conv3_1, conv4_1, conv5_1])

    # Input
    content_maps = tf.constant(content_maps_out, dtype='float32')
    style_maps = [tf.constant(gram_matrix(style_maps_out[i]), dtype='float32') for i in range(len(style_maps_out))]

    # def _cal_squaredNM(m):
    #    m_shape = m.get_shape().as_list()
    #    return 4*(m_shape[0]*m_shape[1])**2

    style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    img_styles = [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]

    def mse(x, y):
        return tf.losses.mean_squared_error(labels=y, predictions=x)

    loss_content = mse(conv4_2, content_maps)
    loss_style = (style_weights[0] * mse(gram_matrix(conv1_1), style_maps[0]) + \
                  style_weights[1] * mse(gram_matrix(conv2_1), style_maps[1]) + \
                  style_weights[2] * mse(gram_matrix(conv3_1), style_maps[2]) + \
                  style_weights[3] * mse(gram_matrix(conv4_1), style_maps[3]) + \
                  style_weights[4] * mse(gram_matrix(conv5_1), style_maps[4]))
    loss = args.content_weight * loss_content + args.style_weight * loss_style

    # Train
    opt = tf.train.AdamOptimizer(args.lr_rate).minimize(loss, var_list=[vgg_input])
    sess.run(tf.global_variables_initializer())
    sess.run(vgg_input.assign(input_img))
    for ep in range(args.epoch + 1):
        _, cur_loss, s_loss, c_loss, img = sess.run([opt, loss, loss_style, loss_content, vgg_input])
        if ep % 50 == 0:
            print('[*] Epoch %d  total_loss=%f, style_loss=%f, content_loss=%f' % (ep, cur_loss, s_loss, c_loss))
        if ep % 500 == 0:
            saved_img = np.array(img[0])
            saved_img = np.where(saved_img <= 255, saved_img, 255)
            saved_img = np.where(saved_img >= 0, saved_img, 0)
            saved_img = Image.fromarray(saved_img.astype(np.uint8), 'RGB')
            output_name = args.output + '_%d' % (ep) + '.jpg'
            saved_img.save(output_name)
            print("[!] image saved as %s\n" % output_name)


if __name__ == '__main__':
    main()