import tensorflow as tf
import numpy as np
from PIL import Image
import os

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

def down_sample(content, style, max_pixel):
    # images are down-sampled according to content image. maximum dimension is equal to or less than specified size
    temp = Image.open(content)
    width, height = temp.size
    max_dim = max(temp.size)
    dsamp_ratio = max_dim / max_pixel
    width = int(width // dsamp_ratio)
    height = int(height // dsamp_ratio)
    content_img = Image.open(content).resize((width, height), resample=Image.LANCZOS)
    style_img = Image.open(style).resize((width, height), resample=Image.LANCZOS)
    return content_img, style_img, height, width


def make_output_folder(content_dir, style_dir, output_name):
    # get output path directory from content directory, style directory, and output folder prefix

    content_dir = os.path.basename(content_dir)
    content_dir, _ = os.path.splitext(content_dir)
    style_dir = os.path.basename(style_dir)
    style_dir, _ = os.path.splitext(style_dir)
    folder_name = "%s_%s" % (style_dir, content_dir)
    full_name = os.path.join(output_name, folder_name)
    if not os.path.exists(full_name):
        os.makedirs(full_name)
    return full_name