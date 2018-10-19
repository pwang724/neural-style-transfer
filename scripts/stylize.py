import tensorflow as tf
import numpy as np
import argparse
import scipy.misc
from PIL import Image
from copy import copy
import model
import utils
import os
import loss

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='../contents/peter.jpg', help='content path')
    parser.add_argument('--style', type=str, default='../styles/mnist.png', help='style path')
    parser.add_argument('--model', type=str, default="C:/Users/Peter/Desktop/models/vgg19.npy", help='vgg19 path')
    parser.add_argument('--output_folder', type=str, default='../out', help='output folder prefix name')
    parser.add_argument('--init_image', type=str, default='content', help='init image: style, content, noise, zeros')
    parser.add_argument('--max_pixel', type=int, default= 400, help='max pixels in any dim')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=3000, help='epoch')
    parser.add_argument('--alpha', type=float, default=1, help='content weight')
    parser.add_argument('--beta', type=float, default=1000000, help='style weight')
    return parser

if __name__=='__main__':
    parser = arg_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    #make output directory
    folder_name = utils.make_output_folder(args_dict['content'], args_dict['style'], args_dict['output_folder'])

    #down-sample image
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
    content_output = [conv4_2]
    style_output = [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
    sess.run(vgg_input.assign(imgs_reshaped['content']))
    content_maps_np, = sess.run([content_output])
    sess.run(vgg_input.assign(imgs_reshaped['style']))
    style_maps_np = sess.run(style_output)
    content_target = [tf.constant(content_maps_np[i], dtype='float32') for i in range(len(content_maps_np))]
    style_target = [tf.constant(utils.gram_matrix(style_maps_np[i]), dtype='float32') for i in range(len(style_maps_np))]

    loss_content, loss_style = loss.losses(content_target, content_output, args_dict['alpha'], style_target, style_output, args_dict['beta'])
    loss = loss_content + loss_style

    #train
    opt = tf.train.AdamOptimizer(args_dict['lr']).minimize(loss, var_list=[vgg_input])

    im = args_dict['init_image']
    if im == 'style':
        input = copy(imgs_reshaped['style'])
    elif im == 'content':
        input = copy(imgs_reshaped['content'])
    elif im == 'zeros':
        input = np.zeros_like(imgs_reshaped['content'])
    else:
        input = np.random.randn(*imgs_reshaped['content'].shape)

    sess.run(tf.global_variables_initializer())
    sess.run(vgg_input.assign(input))

    for ep in range(args.epoch + 1):
        _, cur_loss, s_loss, c_loss, img = sess.run([opt, loss, loss_style, loss_content, vgg_input])
        if ep % 50 == 0:
            print('Epoch %d  total_loss=%.2f, style_loss=%.2f, content_loss=%.2f' % (ep, cur_loss, s_loss, c_loss))
        if ep % 500 == 0:
            saved_img = np.array(img[0])
            saved_img = np.where(saved_img <= 255, saved_img, 255)
            saved_img = np.where(saved_img >= 0, saved_img, 0)
            saved_img = Image.fromarray(saved_img.astype(np.uint8), 'RGB')
            output_name = 'out_%d' % (ep) + '.jpg'
            output_name = os.path.join(folder_name, output_name)
            saved_img.save(output_name)
            print("Saved as %s\n" % output_name)

    print('done')
