from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

def coordinateAttentionLayer3d(x, inputChannel, outputChannel, reductionRatio=16):
    def h_swish(x):
        re_lu = tf.nn.relu6(x+3)/6
        return re_lu * x

    #Hope input x has a shape of NHWC
    identity = x
    [n, h, w, d, c] = x.shape
    x_h = AveragePooling3D(pool_size=(1, w, d), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_w = AveragePooling3D(pool_size=(h, 1, d), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_d = AveragePooling3D(pool_size=(h, w, 1), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_h = Permute((2, 1, 3, 4))(x_h)
    x_d = Permute((1, 3, 2, 4))(x_d)
    y = K.concatenate((x_h, x_w, x_d), axis=2)
    reductionChannel = max(8, inputChannel//reductionRatio)
    y = Conv3D(filters=reductionChannel, kernel_size=1,
               strides=1, padding="valid")(y)
    y = LayerNormalization()(y)
    y = h_swish(y)
    x_h, x_w, x_d = Lambda(tf.split, arguments={"axis": 2, "num_or_size_splits": [w, h, d]})(y)
    x_h = Permute((2, 1, 3, 4))(x_h)
    x_d = Permute((1, 3, 2, 4))(x_d)
    a_h = Conv3D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_h)
    a_w = Conv3D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_w)
    a_d = Conv3D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_d)
    # a_h = tf.tile(a_h, [1, h, 1, 1, 1])
    # a_w = tf.tile(a_w, [1, 1, w, 1, 1])
    # a_d = tf.tile(a_d, [1, 1, 1, d, 1])
    out = multiply([identity, a_w, a_h, a_d])
    return out
