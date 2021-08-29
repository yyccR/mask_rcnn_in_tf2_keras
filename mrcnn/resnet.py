
import tensorflow as tf


class Resnet:
    def __init__(self):
        pass

    def identity_block(self, input_tensor, kernel_size, filters, stage, block,
                       use_bias=True, train_bn=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(filters=nb_filter1,
                                   kernel_size=(1, 1),
                                   name=conv_name_base + '2a',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=nb_filter2,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding='same',
                                   name=conv_name_base + '2b',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=nb_filter3,
                                   kernel_size=(1, 1),
                                   name=conv_name_base + '2c',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.keras.layers.ReLU(name='res' + str(stage) + block + '_out')(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block,
                   strides=(2, 2), use_bias=True, train_bn=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(filters=nb_filter1,
                                   kernel_size=(1, 1),
                                   strides=strides,
                                   name=conv_name_base + '2a',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=nb_filter2,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding='same',
                                   name=conv_name_base + '2b',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=nb_filter3,
                                   kernel_size=(1, 1),
                                   name=conv_name_base + '2c',
                                   use_bias=use_bias,
                                   # kernel_regularizer='l2'
                                   )(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
        shortcut = tf.keras.layers.Conv2D(filters=nb_filter3,
                                          kernel_size=(1, 1),
                                          strides=strides,
                                          name=conv_name_base + '1',
                                          use_bias=use_bias,
                                          # kernel_regularizer='l2'
                                          )(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU(name='res' + str(stage) + block + '_out')(x)
        return x

    def resnet_graph(self, input_image, architecture, stage5=True, train_bn=True):
        """Build a ResNet graph.
            architecture: Can be resnet50 or resnet101
            stage5: Boolean. If False, stage5 of the network is not created
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        x = tf.keras.layers.ZeroPadding2D((3, 3))(input_image)
        # x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True, kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x, training=train_bn)
        # x = KL.Activation('relu')(x)
        x = tf.keras.layers.ReLU()(x)
        C1 = x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
        # Stage 3
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]
        # return C4

    def build_graph(self):
        inputs = tf.keras.layers.Input(shape=[640, 640, 3])
        outputs = self.resnet_graph(inputs, 'resnet50')
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def build_graph_top_down_layer(self):
        # C1 = tf.keras.layers.Input(shape=[320,320,3])
        C2 = tf.keras.layers.Input(shape=[320,320,3],name='C2')
        C3 = tf.keras.layers.Input(shape=[160,160,3],name='C3')
        C4 = tf.keras.layers.Input(shape=[80,80,3],name='C4')
        C5 = tf.keras.layers.Input(shape=[40,40,3],name='C5')

        P5 = tf.keras.layers.Conv2D(256, (1, 1), name='C5_conv')(C5)
        P4 = tf.keras.layers.Add(name="P5_add_C4")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="p5_up_sampled")(P5),
            tf.keras.layers.Conv2D(256, (1, 1), name='C4_conv')(C4)])
        P3 = tf.keras.layers.Add(name="P4_add_C3")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="P4_up_sampled")(P4),
            tf.keras.layers.Conv2D(256, (1, 1), name='C3_conv')(C3)])
        P2 = tf.keras.layers.Add(name="P3_add_C2")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="P3_up_sampled")(P3),
            tf.keras.layers.Conv2D(256, (1, 1), name='C2_conv')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="P2_conv")(P2)
        P3 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="P3_conv")(P3)
        P4 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="P4_conv")(P4)
        P5 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="P5_conv")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, name="P6_pool")(P5)
        # P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, name="fpn_p6")(C5)
        #
        # # Note that P6 is used in RPN, but not in the classifier heads.
        # # [1/4, 1/8, 1/16, 1/32, 1/64]
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        # inputs = tf.keras.layers.Input(shape=[640, 640, 3])
        # outputs = self.resnet_graph(inputs, 'resnet50')
        return tf.keras.models.Model(inputs=[C2,C3,C4,C5], outputs=[rpn_feature_maps, mrcnn_feature_maps])


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    resnet = Resnet()
    resnet_model = resnet.build_graph_top_down_layer()
    resnet_model.summary()
    # tf.keras.utils.plot_model(resnet_model)
    from tensorflow.python.ops import summary_ops_v2
    from tensorflow.python.keras.backend import get_graph
    tb_writer = tf.summary.create_file_writer('./logs')
    with tb_writer.as_default():
        if not resnet_model.run_eagerly:
            summary_ops_v2.graph(get_graph(), step=0)
