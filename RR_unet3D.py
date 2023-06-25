from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
import numpy as np
from New_method import EHCM,DRAM
from attention_code import coordinateAttentionLayer3d
from recurrent_residual_3d import RRCNN_block

class UNet3D(Model):
# 改进的UNet 3D网络
    def __init__(self,
                 n_classes,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="relu",
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=False,
                 logger=None, **kwargs):

        self.img_shape = (dim, dim, dim, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field，计算感受野
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("UpSampling3D")
        # Log the model definition
        self.log()

    def init_model(self):
        """
        Build the UNet model with the specified input image shape.
        """
        inputs = Input(shape=self.img_shape)

        # Apply regularization if not None or 0
        kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        """
        Encoding path
        """
        in_ = inputs
        in_ = Conv3D(64, (7, 7, 7),
                     activation=self.activation, padding=self.padding,
                     kernel_regularizer=kr)(in_)
        in_ = coordinateAttentionLayer3d(in_, 64, 64)
        residual_connections = []
        for i in range(self.depth):
            filters = [64, 128, 256, 512]
            if i == 3:
                bn = RRCNN_block(filters[i])(in_)
            else:
                bn = RRCNN_block(filters[i])(in_)
            # in_ = MaxPooling3D(pool_size=(2, 2, 2))(bn)
            in_ = Conv3D(filters[i], (2, 2, 2), strides=2, activation=self.activation, padding='valid')(bn)

            # Update filter count and add bn layer to list for residual conn.
            residual_connections.append(bn)

        """
        Bottom (no max-pool)
        """
        # bn = aspp(in_, 512)
        # bn = idcm(in_, 512)
        bn = EHCM(in_, 512)
        # conv = Conv3D(512, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(conv)


        """
        Up-sampling
        """

        residual_connections = residual_connections[::-1]  # 倒向输出bn3，bn2，bn1
        # code 1
        bn = LayerNormalization()(bn)
        bn10 = UpSampling3D(size=(2, 2, 2))(bn)
        bn111 = Conv3D(512, (1, 1, 1),
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(bn10)
        bn111 = DRAM(bn111, residual_connections[0], 512)
        # bn12 = Conv3D(512, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn111)
        # bn13 = Conv3D(512, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn12)
        bn1 = RRCNN_block(512)(bn111)
        # bn1 = Add()([bn13, bn111])
        # bn1 = channel_spatial_squeeze_excite(bn1)
        # bn1 = cbam_block(bn1)
        # code 2
        bn1 = LayerNormalization()(bn1)
        bn20 = UpSampling3D(size=(2, 2, 2))(bn1)


        # bn21 = channel_spatial_squeeze_excite(bn21)
        bn222 = Conv3D(256, (1, 1, 1),
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(bn20)

        bn222 = DRAM(bn222, residual_connections[1], 256)
        bn2 = RRCNN_block(256)(bn222)
        # bn22 = Conv3D(256, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn222)
        # bn23 = Conv3D(256, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn22)
        # bn2 = Add()([bn23, bn222])

        # code 3
        bn2 = LayerNormalization()(bn2)
        bn30 = UpSampling3D(size=(2, 2, 2))(bn2)
        # bn31 = channel_spatial_squeeze_excite(bn31)
        bn333 = Conv3D(128, (1, 1, 1),
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(bn30)
        bn333 = DRAM(bn333, residual_connections[2], 128)
        # bn32 = Conv3D(128, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn333)
        # bn33 = Conv3D(128, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn32)
        # bn3 = Add()([bn33, bn333])
        bn3 = RRCNN_block(128)(bn333)
        # code 4
        bn3 = LayerNormalization()(bn3)
        bn40 = UpSampling3D(size=(2, 2, 2))(bn3)
        # # bn41 = channel_spatial_squeeze_excite(bn41)
        bn444 = Conv3D(64, (1, 1, 1),
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(bn40)
        bn444 = DRAM(bn444, residual_connections[3], 64)
        # bn42 = Conv3D(64, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn444)
        # bn43 = Conv3D(64, kernel_size=3,
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(bn42)
        # bn4 = Add()([bn43, bn444])
        bn4= RRCNN_block(64)(bn444)
        """
        Output modeling layer and deep supervised learning
        """
        # add1 = Add()([bn222, bn2])
        # add1 = LayerNormalization()(add1)
        # add1 = UpSampling3D(size=(2, 2, 2))(add1)
        # add1 = Conv3D(128, (1, 1, 1),
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(add1)
        # add2 = Add()([add1, bn3])
        # add2 = UpSampling3D(size=(2, 2, 2))(add2)
        # add2 = Conv3D(64, (1, 1, 1),
        #               activation=self.activation, padding=self.padding,
        #               kernel_regularizer=kr)(add2)
        # add3 = Add()([add2, bn4])
        out = Conv3D(self.n_classes, 1, activation=self.out_activation)(bn4)
        # out = Conv3D(self.n_classes, 1, activation=self.out_activation)(bn4)
        if self.flatten_output:
            out = Reshape([np.prod(self.img_shape[:3]),
                           self.n_classes], name='flatten_output')(out)

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping3D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]  # 第二个到倒数第二个,也就是除开batch和channel，长宽高
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping3D(cr)(node1)  # 对三个维度里面每个维度的头尾都裁剪cr形状
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

