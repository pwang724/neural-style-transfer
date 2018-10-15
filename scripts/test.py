import tensorflow as tf
import numpy as np
import argparse
import scipy.misc
from PIL import Image
from copy import copy
import model
import utils

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='../contents/bill.jpg', help='content path')
    parser.add_argument('--style', type=str, default='../styles/starry.jpg', help='style path')
    parser.add_argument('--model', type=str, default="C:/Users/Peter/Desktop/models/vgg19.npy", help='vgg19 path')
    parser.add_argument('--output', type=str, default='out', help='output folder')
    parser.add_argument('--max_pixel', type=int, default= 400, help='max pixels in any dim')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=3000, help='epoch')
    parser.add_argument('--alpha', type=float, default=1, help='content weight')
    parser.add_argument('--beta', type=float, default=100000, help='style weight')
    return parser

def losses(content_maps, style_maps, style_output):
    def mse(x, y):
        return tf.losses.mean_squared_error(labels=y, predictions=x)

    style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    style_weights = tf.unstack(tf.constant(style_weights), axis= 0)

    loss_content = mse(conv4_2, content_maps)
    loss_style_raw = [mse(utils.gram_matrix(pred), label) for label, pred in zip(style_maps, style_output)]
    loss_style = [tf.multiply(weight, layer_loss) for weight, layer_loss in zip(style_weights, loss_style_raw)]
    loss_style_red = tf.reduce_sum(loss_style)
    return loss_content, loss_style_red

parser = arg_parser()
args = parser.parse_args()
args_dict = vars(args)

content, style, height, width = utils.down_sample(args_dict['content'], args_dict['style'], args_dict['max_pixel'])
assert content.mode == 'RGB', 'content image not in RGB format'
assert style.mode == 'RGB', 'style image not in RGB format'

# input tensor: input image with shape of [batch, height, width, colors=3]
f_img_reshape = lambda x: np.reshape(np.asarray(x), newshape=(-1, height, width, 3))
imgs = {'content':content , 'style': style}
imgs_reshaped = {key: f_img_reshape(img) for key, img in imgs.items()}
vgg_input = tf.Variable(initial_value=np.zeros(shape=[1, height, width, 3], dtype='float32'), name='image')

#build model
conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2 = model.build_part_vgg19(vgg_input, args_dict['model'])

#gpu
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

#get content and style loss
sess.run(vgg_input.assign(imgs_reshaped['content']))
content_maps_np, = sess.run([conv4_2])
sess.run(vgg_input.assign(imgs_reshaped['style']))
style_maps_np = sess.run([conv1_1, conv2_1, conv3_1, conv4_1, conv5_1])

content_maps = tf.constant(content_maps_np, dtype='float32')
style_maps = [tf.constant(utils.gram_matrix(style_maps_np[i]), dtype='float32') for i in range(len(style_maps_np))]
network_style_output = [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]

#loss
loss_content, loss_style = losses(content_maps, style_maps, network_style_output)
loss = args_dict['alpha'] * loss_content + args_dict['beta'] * loss_style

#train
opt = tf.train.AdamOptimizer(args_dict['lr']).minimize(loss, var_list=[vgg_input])
input = copy(imgs_reshaped['content'])
sess.run(tf.global_variables_initializer())
sess.run(vgg_input.assign(input))
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


print('done')
