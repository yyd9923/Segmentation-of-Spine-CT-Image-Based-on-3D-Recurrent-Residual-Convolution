import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
#高效高效密集连接混合卷积模块
def EHCM(inputs, filters):

        x1 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='SAME', dilation_rate=(1, 1, 1))(inputs)
        x2 = Concatenate(axis=-1)([inputs, x1])

        x3 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='SAME', dilation_rate=(2, 2, 2))(x2)
        x4 = Concatenate(axis=-1)([x2, x3])

        x5 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='SAME', dilation_rate=(5, 5, 5))(x4)
        x6 = Concatenate(axis=-1)([x4, x5])

        x7 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='same', use_bias=False)(x6)
        x = LayerNormalization()(x7)
        x = Activation('relu')(x)
        return x



#双特征残差注意力机制
def DRAM(input1, input2, filters):
    x1 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='same', dilation_rate=(2, 2, 2))(input1)
    x1 = Add()([x1, input1])
    x2 = Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='same', dilation_rate=(2, 2, 2))(input2)
    x2 = Add()([x2, input2])
    x11 = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, padding='same')(x1)
    x22 = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, padding='same')(x2)
    add = Concatenate(axis=-1)([x11, x22])
    add = Activation("relu")(add)
    add = Conv3D(filters, kernel_size=(1, 1, 1), strides=1, padding='same')(add)
    add = Activation("sigmoid")(add)
    x = Add()([input2, input1])
    output = multiply([add, x])
    return output



