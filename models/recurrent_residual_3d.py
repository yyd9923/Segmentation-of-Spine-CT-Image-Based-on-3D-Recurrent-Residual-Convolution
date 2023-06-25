import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

class Recurrent_block(Model):
    """
    三维残差循环卷积层
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = Sequential([
            Conv3D(out_ch, kernel_size=(3, 3, 3), strides=1, padding='same'),
            LayerNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out

class RRCNN_block(Model):


    def __init__(self, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = Sequential([
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        ])
        self.Conv = Conv3D(out_ch, kernel_size=(1, 1, 1), strides=1, padding='same')

    def call(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out
