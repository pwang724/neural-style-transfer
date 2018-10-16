import tensorflow as tf
import utils

def losses(content_maps, content_output, content_weight, style_maps, style_output, style_weight):
    def mse(y, x):
        return tf.losses.mean_squared_error(labels=y, predictions=x)
    #TODO: content weights
    loss_content = [mse(pred, label) for label, pred in zip(content_maps, content_output)]
    loss_content_red = tf.reduce_sum(loss_content)
    loss_content_red *= content_weight

    style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    style_weights = tf.unstack(tf.constant(style_weights), axis= 0)
    loss_style_raw = [mse(utils.gram_matrix(pred), label) for label, pred in zip(style_maps, style_output)]
    loss_style = [tf.multiply(weight, layer_loss) for weight, layer_loss in zip(style_weights, loss_style_raw)]
    loss_style_red = tf.reduce_sum(loss_style)
    loss_style_red *= style_weight
    return loss_content_red, loss_style_red