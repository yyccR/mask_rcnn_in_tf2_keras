import sys

sys.path.append("../../mask_rcnn_in_tf2_keras")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from mrcnn.resnet import Resnet
from mrcnn.anchors_ops import get_anchors, AnchorsLayer
from mrcnn.bbox_ops import norm_boxes_graph, norm_boxes, denorm_boxes, unmold_mask, clip_boxes_graph
from mrcnn.layers import DetectionTargetLayer, ProposalLayer, FPNMaskLayer, PyramidROIAlignLayer
from mrcnn.layers import FPNClassifyLayer, build_rpn_targets, DetectionLayer, DetectionMaskLayer
from data.generate_voc_segment_data import VocSegmentDataGenerator
from mrcnn.losses import rpn_bbox_loss, rpn_class_loss
from mrcnn.losses import mrcnn_mask_loss, mrcnn_class_loss, mrcnn_bbox_loss
from data.visual_ops import draw_bounding_box, draw_instance
from data.generate_tfrecord_files import parse_voc_segment_tfrecord


class MaskRCNN:
    def __init__(self,
                 classes,
                 image_shape=[640, 640, 3],
                 batch_size=2,
                 is_training=True,
                 scales=[32, 64, 128, 256, 512],
                 # scales=[32, 128, 128, 128, 256],
                 ratios=[0.5, 1, 2],
                 feature_strides=[4, 8, 16, 32, 64],
                 anchor_stride=1,
                 post_nms_rois_training=2000,
                 post_nms_rois_inference=1000,
                 rpn_nms_threshold=0.7,
                 rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                 rpn_nms_limit=6000,
                 rpn_train_anchors_per_image=256,
                 train_rois_per_image=200,
                 roi_positive_ratio=0.33,
                 bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                 mask_shape=[28, 28],
                 use_mini_mask=True,
                 mini_mask_shape=(56, 56),
                 fpn_pool_size=[7, 7],
                 mask_pool_size=[14, 14],
                 fpn_fc_layers_size=1024,
                 detection_min_confidence=0.3,
                 detection_nms_thres=0.3,
                 detection_max_instances=100,
                 pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])):
        self.classes = classes
        self.num_classes = len(classes)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.is_training = is_training
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
        self.anchor_stride = anchor_stride
        self.anchors_per_location = len(self.ratios)
        self.post_nms_rois_training = post_nms_rois_training
        self.post_nms_rois_inference = post_nms_rois_inference
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.rpn_nms_limit = rpn_nms_limit
        self.rpn_train_anchors_per_image = rpn_train_anchors_per_image
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.mask_shape = mask_shape
        self.mini_mask_shape = mini_mask_shape
        self.fpn_pool_size = fpn_pool_size
        self.mask_pool_size = mask_pool_size
        self.fpn_fc_layers_size = fpn_fc_layers_size
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_thres = detection_nms_thres
        self.detection_max_instances = detection_max_instances
        self.pixel_mean = pixel_mean

        self.base_model = Resnet()
        self.mrcnn = self.build_graph(is_training=is_training)

    def _backbone(self, input_images):
        # [1/4, 1/4, 1/8, 1/16, 1/32]
        # C4 = self.base_model.resnet_graph(input_images, 'resnet50')
        # C4 = self.base_model.image_to_head(input_images)
        C1, C2, C3, C4, C5 = self.base_model.resnet_graph(input_images, 'resnet101')

        P5 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = tf.keras.layers.Add(name="fpn_p4add")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = tf.keras.layers.Add(name="fpn_p3add")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = tf.keras.layers.Add(name="fpn_p2add")([
            tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = tf.keras.layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        # P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, name="fpn_p6")(C5)
        #
        # # Note that P6 is used in RPN, but not in the classifier heads.
        # # [1/4, 1/8, 1/16, 1/32, 1/64]
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        # rpn_feature_maps = [C4]
        # mrcnn_feature_maps = C4

        return rpn_feature_maps, mrcnn_feature_maps

    def rpn_graph(self, feature_map, anchors_per_location, anchor_stride, fpn_level):
        """ """
        # TODO: check if stride of 2 causes alignment issues if the feature map
        shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                                        name='rpn_conv_shared_' + fpn_level)(feature_map)
        x = tf.keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                                   activation='linear',
                                   # name='rpn_class_raw_' + fpn_level)(shared)
                                   name='rpn_class_raw_' + fpn_level)(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])

        # Softmax on last dimension of BG/FG.
        rpn_probs = tf.keras.layers.Softmax(name="rpn_class_xxx_" + fpn_level)(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = tf.keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                                   activation='linear',
                                   # name='rpn_bbox_pred_' + fpn_level)(shared)
                                   name='rpn_bbox_pred_' + fpn_level)(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox_delta = tf.reshape(x, [tf.shape(x)[0], -1, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox_delta]

    def fpn_classify(self, rois, mrcnn_feature_maps, is_training):
        """ mask-rcnn 分类,边框预测层
        :param rois:
        :param mrcnn_feature_maps:
        :return:
        """
        pooled = PyramidROIAlignLayer(
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            pool_shape=self.fpn_pool_size
        )(rois, mrcnn_feature_maps)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.fpn_fc_layers_size,
                                   kernel_size=self.fpn_pool_size),
            name="mrcnn_class_conv1"
        )(pooled)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.fpn_fc_layers_size,
                                   kernel_size=(1, 1)),
            name="mrcnn_class_conv2"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
        shared = tf.squeeze(tf.squeeze(x, axis=3), axis=2)

        # 类别预测
        mrcnn_class_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_classes),
            name='mrcnn_class_logits'
        )(shared)
        mrcnn_class = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Softmax(),
            name="mrcnn_class"
        )(mrcnn_class_logits)
        # 边框预测
        mrcnn_bbox = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_classes * 4,
                                  activation='linear'),
            name='mrcnn_bbox_fc'
        )(shared)
        mrcnn_bbox = tf.reshape(
            mrcnn_bbox,
            (tf.shape(mrcnn_bbox)[0], tf.shape(mrcnn_bbox)[1], self.num_classes, 4),
            name="mrcnn_bbox")

        return mrcnn_class_logits, mrcnn_class, mrcnn_bbox

    def fpn_mask_predict(self, rois, mrcnn_feature_maps, is_training):
        """ mask-rcnn mask预测层
        :param rois:
        :param mrcnn_feature_maps:
        :return:
        """
        pooled = PyramidROIAlignLayer(
            image_shape=self.image_shape,
            batch_size=self.batch_size,
            pool_shape=self.mask_pool_size
        )(rois, mrcnn_feature_maps)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv1"
        )(pooled)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv2"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv3"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv4"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        # [batch, num_rois,  pool_size*2, pool_size*2, channels]
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(filters=256,
                                            kernel_size=(2, 2),
                                            strides=2,
                                            activation="relu"),
            name="mrcnn_mask_deconv"
        )(x)

        # [batch, num_rois, pool_size*2, pool_size*2, num_classes]
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.num_classes,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   activation="sigmoid"),
            name="mrcnn_mask"
        )(x)
        return x

    def build_graph(self, is_training=True):

        input_images = tf.keras.layers.Input(shape=self.image_shape, batch_size=self.batch_size)

        if is_training:
            input_gt_boxes = tf.keras.layers.Input(
                shape=[None, 4], batch_size=self.batch_size, name="input_gt_boxes", dtype=tf.float32)
            # 对box归一化
            gt_boxes = tf.keras.layers.Lambda(lambda x: norm_boxes_graph(x, self.image_shape[0:2]))(input_gt_boxes)

            # rpn类别目标值以及box目标值
            # rpn_target_match = tf.keras.layers.Input(shape=[None], batch_size=self.batch_size, dtype=tf.float32)
            # rpn_target_box = tf.keras.layers.Input(shape=[None, 4], batch_size=self.batch_size, dtype=tf.float32)

            if self.use_mini_mask:
                gt_masks = tf.keras.layers.Input(
                    shape=[self.mini_mask_shape[0], self.mini_mask_shape[1], None],
                    batch_size=self.batch_size,
                    name="input_gt_masks",
                    dtype=tf.int8)
            else:
                gt_masks = tf.keras.layers.Input(
                    shape=[self.image_shape[0], self.image_shape[1], None],
                    batch_size=self.batch_size,
                    name="input_gt_masks",
                    dtype=tf.int8)
            gt_classes = tf.keras.layers.Input([None], batch_size=self.batch_size, dtype=tf.int8)
            all_anchors = tf.keras.layers.Input([None, 4], batch_size=self.batch_size, name="input_anchors")
            anchors = all_anchors[0]

        else:
            all_anchors = tf.keras.layers.Input([None, 4], batch_size=self.batch_size, name="input_anchors")
            anchors = all_anchors[0]

        # 前卷积, 拿到各个特征层
        rpn_feature_maps, mrcnn_feature_maps = self._backbone(input_images=input_images)
        layer_outputs = []
        # rpn对每个特征层提取边框和类别
        for i, p in enumerate(rpn_feature_maps):
            layer_outputs.append(self.rpn_graph(feature_map=p,
                                                anchors_per_location=self.anchors_per_location,
                                                anchor_stride=self.anchor_stride,
                                                fpn_level=str(i)))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox_delta"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
        # print(outputs)
        rpn_class_logits, rpn_class, rpn_bbox_delta = outputs
        # rpn_class_logits, rpn_class, rpn_bbox_delta = layer_outputs[0]

        if is_training:
            # proposal对score排序和采样,再将预测的box delta映射到对应的anchor上
            proposal_count = self.post_nms_rois_training
            rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=self.rpn_nms_threshold,
                rpn_bbox_std_dev=self.rpn_bbox_std_dev,
                rpn_nms_limit=self.rpn_nms_limit,
                batch_size=self.batch_size)(rpn_class, rpn_bbox_delta, anchors)

            # detect target把proposal输出的roi和gt_box做偏差计算, 同时筛选出指定数量的样本和对应的目标, 作为损失计算用
            rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask = DetectionTargetLayer(
                batch_size=self.batch_size,
                train_rois_per_image=self.train_rois_per_image,
                roi_positive_ratio=self.roi_positive_ratio,
                bbox_std_dev=self.bbox_std_dev,
                use_mini_mask=self.use_mini_mask,
                mask_shape=self.mask_shape
            )(rpn_rois, gt_classes, gt_boxes, gt_masks)

            # mask rcnn网络预测最终类别和边框
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classify(rois, mrcnn_feature_maps, is_training)
            # mask rcnn网络预测最终mask
            mrcnn_mask = self.fpn_mask_predict(rois, mrcnn_feature_maps, is_training)

            # Model
            inputs = [input_images, input_gt_boxes, gt_classes, gt_masks, all_anchors]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox_delta,
                       rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask]
            model = tf.keras.models.Model(inputs, outputs, name='mask_rcnn')
            return model

        else:
            # proposal对score排序和采样,再将预测的box delta映射到对应的anchor上
            proposal_count = self.post_nms_rois_inference
            rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=self.rpn_nms_threshold,
                rpn_bbox_std_dev=self.rpn_bbox_std_dev,
                rpn_nms_limit=self.rpn_nms_limit,
                batch_size=self.batch_size)(rpn_class, rpn_bbox_delta, anchors)

            # mask rcnn网络预测最终类别和边框
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classify(rpn_rois, mrcnn_feature_maps, is_training)

            # 利用rpn输出的roi和mask rcnn输出的类别和边框得到最终的box和class
            # [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            detections = DetectionLayer(
                batch_size=self.batch_size,
                bbox_std_dev=self.bbox_std_dev,
                detection_max_instances=self.detection_max_instances,
                detection_nms_thres=self.detection_nms_thres,
                detection_min_confidence=self.detection_min_confidence
            )(rpn_rois, mrcnn_class, mrcnn_bbox,
              window=np.array([0, 0, self.image_shape[0], self.image_shape[1]], dtype=np.float32))

            # mask rcnn网络预测最终mask
            mrcnn_mask = self.fpn_mask_predict(detections[..., :4], mrcnn_feature_maps, is_training)

            model = tf.keras.Model([input_images, all_anchors],
                                   [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask],
                                   name='mask_rcnn')
            return model

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape):
        """ 将detection中的box和class跟 mask-rcnn预测的mask做提取转换

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        detections = np.array(detections)
        mrcnn_mask = np.array(mrcnn_mask)
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        # window = norm_boxes(original_window, self.image_shape[:2])
        # wy1, wx1, wy2, wx2 = window
        # shift = np.array([wy1, wx1, wy1, wx1])
        # wh = wy2 - wy1  # window height
        # ww = wx2 - wx1  # window width
        # scale = np.array([wh, ww, wh, ww])
        # # Convert boxes to normalized coordinates on the window
        # boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = denorm_boxes(boxes, original_image_shape[:2])
        boxes = clip_boxes_graph(boxes, np.array([0, 0, self.image_shape[0], self.image_shape[1]], dtype=np.int32))

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((original_image_shape[0], original_image_shape[1], 0))

        return boxes, class_ids, scores, full_masks

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            layers = self.mrcnn.inner_model.layers if hasattr(self.mrcnn, "inner_model") \
                else self.mrcnn.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

    def train(self, epochs, log_dir, data_path):
        """ 边生成数据边训练, 不用tf record格式

        :param epochs
        :param log_dir
        :param tfrec_path: tfrecord 数据路径
        """

        vsg_train = VocSegmentDataGenerator(
            voc_data_path=data_path,
            batch_size=self.batch_size,
            class_balance=True,
            is_training=self.is_training,
            im_size=self.image_shape[0],
            data_max_size_per_class=2
        )

        mrcnn = self.mrcnn
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        anchors = get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              anchor_stride=self.anchor_stride)
        # all_anchors = np.stack([anchors, anchors], axis=0)
        all_anchors = np.tile([anchors], reps=[self.batch_size, 1, 1])
        # tensorboard 日志目录
        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(epochs):
            for batch in range(vsg_train.total_batch_size):
                imgs, masks, gt_boxes, labels = vsg_train.next_batch()
                rpn_target_match, rpn_target_box = build_rpn_targets(
                    anchors=anchors,
                    gt_boxes=gt_boxes,
                    image_shape=self.image_shape[0:2],
                    batch_size=self.batch_size,
                    rpn_train_anchors_per_image=self.rpn_train_anchors_per_image,
                    rpn_bbox_std_dev=self.rpn_bbox_std_dev)

                print(np.shape(imgs))
                print(np.shape(masks))
                print(np.shape(gt_boxes))
                print("-------{}-{}--------".format(epoch, batch))

                if np.sum(gt_boxes) <= 0.:
                    print(batch, " gt_boxes: ", gt_boxes)
                    continue

                if epoch % 20 == 0 and epoch != 0:
                    mrcnn.save_weights("./mrcnn-epoch-{}.h5".format(epoch))

                with tf.GradientTape() as tape:
                    # 模型输出
                    # rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                    rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                    mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask, mrcnn_class_logits, \
                    mrcnn_class, mrcnn_bbox, mrcnn_mask = \
                        mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)
                    # mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)

                    # 计算损失
                    rpn_c_loss = rpn_class_loss(rpn_target_match, rpn_class_logits)
                    rpn_b_loss = rpn_bbox_loss(rpn_target_box, rpn_target_match, rpn_bbox_delta)
                    mrcnn_c_loss = mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_class_logits, rois)
                    mrcnn_b_loss = mrcnn_bbox_loss(mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois)
                    mrcnn_m_bc_loss = mrcnn_mask_loss(mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois)
                    total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_bc_loss

                    # 梯度更新
                    grad = tape.gradient(total_loss, mrcnn.trainable_variables)
                    optimizer.apply_gradients(zip(grad, mrcnn.trainable_variables))

                    # tensorboard 损失曲线
                    with summary_writer.as_default():
                        tf.summary.scalar('loss/rpn_class_loss', rpn_c_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)
                        tf.summary.scalar('loss/rpn_bbox_loss', rpn_b_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_class_loss', mrcnn_c_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_bbox_loss', mrcnn_b_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_mask_binary_crossentropy_loss', mrcnn_m_bc_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)
                        tf.summary.scalar('loss/total_loss', total_loss,
                                          step=epoch * vsg_train.total_batch_size + batch)

                    # 非极大抑制与其他条件过滤
                    # [b, num_detections, (y1, x1, y2, x2, class_id, score)], [b, num_detections, h, w, num_classes]
                    detections, pred_masks = DetectionMaskLayer(
                        batch_size=self.batch_size,
                        bbox_std_dev=self.bbox_std_dev,
                        detection_max_instances=self.detection_max_instances,
                        detection_nms_thres=self.detection_nms_thres,
                        detection_min_confidence=self.detection_min_confidence
                    )(rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, np.array([0, 0, 1, 1], np.float32))

                    for i in range(self.batch_size):
                        # 将数据处理成原图大小
                        boxes, class_ids, scores, full_masks = self.unmold_detections(
                            detections=detections[i],
                            mrcnn_mask=pred_masks[i],
                            original_image_shape=self.image_shape)

                        # 预测结果
                        pred_img = imgs[i].copy() + self.pixel_mean
                        for j in range(np.shape(class_ids)[0]):
                            score = scores[j]
                            if score > 0.1:
                                class_name = self.classes[class_ids[j]]
                                ymin, xmin, ymax, xmax = boxes[j]
                                pred_mask_j = full_masks[:, :, j]
                                pred_img = draw_instance(pred_img, pred_mask_j)
                                pred_img = draw_bounding_box(pred_img, class_name, score, xmin, ymin, xmax, ymax)

                        # ground true
                        gt_img = imgs[i].copy() + self.pixel_mean
                        active_num = len(np.where(labels[i])[0])
                        for j in range(active_num):
                            l = labels[i][j]
                            class_name = self.classes[l]
                            ymin, xmin, ymax, xmax = gt_boxes[i][j]
                            gt_mask_j = unmold_mask(np.array(masks[i][:, :, j], dtype=np.float32), gt_boxes[i][j],
                                                    self.image_shape)
                            gt_img = draw_bounding_box(gt_img, class_name, l, xmin, ymin, xmax, ymax)
                            gt_img = draw_instance(gt_img, gt_mask_j)

                        concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                        summ_imgs = tf.expand_dims(concat_imgs, 0)
                        summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                        with summary_writer.as_default():
                            tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs, step=batch)

    def train_with_tfrecord(self, epochs, log_dir, tfrec_path):
        """ 预先将数据处理成tfrecord格式，再进行训练，速度可以快很多

        :param tfrec_path: tfrecord 数据路径
        """

        vsg_train = parse_voc_segment_tfrecord(
            is_training=True,
            tfrec_path=tfrec_path,
            repeat=1,
            shuffle_buffer=800,
            batch=self.batch_size
        )
        # tfrecord没法获取数据的size
        total_batch_size = 0
        for _ in vsg_train:
            total_batch_size += 1

        mrcnn = self.mrcnn
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        anchors = get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              anchor_stride=self.anchor_stride)
        all_anchors = np.stack([anchors, anchors], axis=0)
        # tensorboard 日志目录
        summary_writer = tf.summary.create_file_writer(log_dir)

        # for epoch in range(epochs):
        #     for batch in range(vsg_train.total_batch_size):
        #         imgs, masks, gt_boxes, labels = vsg_train.next_batch()
        for epoch in range(epochs):
            batch = 0
            for imgs, masks, gt_boxes, labels, rpn_target_match, rpn_target_box in vsg_train:
                print(np.shape(imgs))
                print(np.shape(masks))
                print(np.shape(gt_boxes))
                print("-------{}--------".format(batch))

                batch += 1
                if np.sum(gt_boxes) <= 0.:
                    print(batch, " gt_boxes: ", gt_boxes)
                    continue

                if epoch % 20 == 0 and epoch != 0:
                    mrcnn.save_weights("./mrcnn-epoch-{}.h5".format(epoch))

                with tf.GradientTape() as tape:
                    # 模型输出
                    # rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                    rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                    mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask, mrcnn_class_logits, \
                    mrcnn_class, mrcnn_bbox, mrcnn_mask = \
                        mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)
                    # mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)

                    # 计算损失
                    rpn_c_loss = rpn_class_loss(rpn_target_match, rpn_class_logits)
                    rpn_b_loss = rpn_bbox_loss(rpn_target_box, rpn_target_match, rpn_bbox_delta)
                    mrcnn_c_loss = mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_class_logits, rois)
                    mrcnn_b_loss = mrcnn_bbox_loss(mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois)
                    mrcnn_m_bc_loss = mrcnn_mask_loss(mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois)
                    total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_bc_loss

                    # 梯度更新
                    grad = tape.gradient(total_loss, mrcnn.trainable_variables)
                    optimizer.apply_gradients(zip(grad, mrcnn.trainable_variables))

                    # tensorboard 损失曲线
                    with summary_writer.as_default():
                        tf.summary.scalar('loss/rpn_class_loss', rpn_c_loss,
                                          step=epoch * total_batch_size + batch)
                        tf.summary.scalar('loss/rpn_bbox_loss', rpn_b_loss,
                                          step=epoch * total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_class_loss', mrcnn_c_loss,
                                          step=epoch * total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_bbox_loss', mrcnn_b_loss,
                                          step=epoch * total_batch_size + batch)
                        tf.summary.scalar('loss/mrcnn_mask_binary_crossentropy_loss', mrcnn_m_bc_loss,
                                          step=epoch * total_batch_size + batch)
                        tf.summary.scalar('loss/total_loss', total_loss,
                                          step=epoch * total_batch_size + batch)

                    # 非极大抑制与其他条件过滤
                    # [b, num_detections, (y1, x1, y2, x2, class_id, score)], [b, num_detections, h, w, num_classes]
                    detections, pred_masks = DetectionMaskLayer(
                        batch_size=self.batch_size,
                        bbox_std_dev=self.bbox_std_dev,
                        detection_max_instances=self.detection_max_instances,
                        detection_nms_thres=self.detection_nms_thres,
                        detection_min_confidence=self.detection_min_confidence
                    )(rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, np.array([0, 0, 1, 1], np.float32))

                    for i in range(self.batch_size):
                        # 将数据处理成原图大小
                        boxes, class_ids, scores, full_masks = self.unmold_detections(
                            detections=detections[i],
                            mrcnn_mask=pred_masks[i],
                            original_image_shape=self.image_shape)

                        # 预测结果
                        pred_img = imgs[i].numpy().copy() + self.pixel_mean
                        for j in range(np.shape(class_ids)[0]):
                            score = scores[j]
                            if score > 0.5:
                                class_name = self.classes[class_ids[j]]
                                ymin, xmin, ymax, xmax = boxes[j]
                                pred_mask_j = full_masks[:, :, j]
                                pred_img = draw_instance(pred_img, pred_mask_j)
                                pred_img = draw_bounding_box(pred_img, class_name, score, xmin, ymin, xmax, ymax)

                        # ground true
                        gt_img = imgs[i].numpy().copy() + self.pixel_mean
                        active_num = len(np.where(labels[i])[0])
                        for j in range(active_num):
                            l = labels[i][j]
                            class_name = self.classes[l]
                            ymin, xmin, ymax, xmax = gt_boxes[i][j]
                            gt_mask_j = unmold_mask(np.array(masks[i][:, :, j],dtype=np.float32), gt_boxes[i][j], self.image_shape)
                            gt_img = draw_bounding_box(gt_img, class_name, l, xmin, ymin, xmax, ymax)
                            gt_img = draw_instance(gt_img, gt_mask_j)

                        concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                        summ_imgs = tf.expand_dims(concat_imgs, 0)
                        summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                        with summary_writer.as_default():
                            tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs, step=batch)

    def test(self, model_path, log_dir, tfrec_path):
        """ 测试评估"""

        self.load_weights(model_path, by_name=True)
        summary_writer = tf.summary.create_file_writer(log_dir)
        vsg_train = parse_voc_segment_tfrecord(
            is_training=True,
            tfrec_path=tfrec_path,
            repeat=1,
            shuffle_buffer=800,
            batch=self.batch_size
        )

        anchors = get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              anchor_stride=self.anchor_stride)
        all_anchors = np.stack([anchors] * self.batch_size, axis=0)

        step = 0
        for imgs, masks, gt_boxes, labels, rpn_target_match, rpn_target_box in vsg_train:
            detections, mrcnn_class, mrcnn_bbox, mrcnn_mask = self.mrcnn.predict([imgs, all_anchors])
            for i in range(self.batch_size):
                step += 1
                boxes, class_ids, scores, full_masks = self.unmold_detections(detections=detections[i],
                                                                              mrcnn_mask=mrcnn_mask[i],
                                                                              original_image_shape=self.image_shape)
                # 预测结果
                pred_img = imgs[i].numpy().copy() + self.pixel_mean
                for j in range(np.shape(class_ids)[0]):
                    score = scores[j]
                    if score > 0.5:
                        class_name = self.classes[class_ids[j]]
                        ymin, xmin, ymax, xmax = boxes[j]
                        pred_mask_j = full_masks[:, :, j]
                        pred_img = draw_instance(pred_img, pred_mask_j)
                        pred_img = draw_bounding_box(pred_img, class_name, score, xmin, ymin, xmax, ymax)

                # ground true
                gt_img = imgs[i].numpy().copy() + self.pixel_mean
                active_num = len(np.where(labels[i])[0])
                for j in range(active_num):
                    l = labels[i][j]
                    class_name = self.classes[l]
                    ymin, xmin, ymax, xmax = gt_boxes[i][j]
                    gt_mask_j = unmold_mask(np.array(masks[i][:,:, j],dtype=np.float32), gt_boxes[i][j], self.image_shape)
                    gt_img = draw_bounding_box(gt_img, class_name, l, xmin, ymin, xmax, ymax)
                    gt_img = draw_instance(gt_img, gt_mask_j)


                concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                summ_imgs = tf.expand_dims(concat_imgs, 0)
                summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                with summary_writer.as_default():
                    tf.summary.image("imgs/gt,pred,epoch{}".format(step // 30), summ_imgs, step=step)

    def predict(self, image, anchors, draw_detect_res_figure=True):
        """ 预测，输入的batch=1, batch跟随模型构建过程

        :param image: [batch, h, w, c]
        :param anchors: [batch, (y1, x1, y2, x2)]

        :return boxes,class_ids,scores,masks
        """
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask = self.mrcnn.predict([image, anchors])
        final_boxes = final_class_ids = final_scores = final_mask = []
        for i in range(self.batch_size):
            # 预测结果, 数据处理回原图大小
            boxes, class_ids, scores, full_masks = self.unmold_detections(detections=detections[i],
                                                                          mrcnn_mask=mrcnn_mask[i],
                                                                          original_image_shape=self.image_shape)
            final_boxes.append(boxes)
            final_class_ids.append(class_ids)
            final_scores.append(scores)
            final_mask.append(full_masks)

            # 检测结果保存图片
            if draw_detect_res_figure:
                pred_img = image[i].numpy().copy() + self.pixel_mean
                for j in range(np.shape(class_ids)[0]):
                    score = scores[j]
                    if score > 0.5:
                        class_name = self.classes[class_ids[j]]
                        ymin, xmin, ymax, xmax = boxes[j]
                        pred_mask_j = full_masks[:, :, j]
                        pred_img = draw_instance(pred_img, pred_mask_j)
                        pred_img = draw_bounding_box(pred_img, class_name, score, xmin, ymin, xmax, ymax)
                        pred_img = np.array(pred_img, dtype=np.uint8)
                        cv2.imwrite("../data/tmp/{}.jpeg".format(i), pred_img)

        return final_boxes, final_class_ids, final_scores, final_mask


if __name__ == "__main__":
    mrcnn = MaskRCNN(classes=['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                     is_training=True,
                     batch_size=1,
                     image_shape=[320,320,3]
                     )
    # mrcnn = MaskRCNN(classes=['_background_', 'aeroplane'],
    #                  voc_data_path='../../data/VOCdevkit/VOC2012/'
    #                  )
    # model = mrcnn.build_graph(is_training=False)
    # model.summary(line_length=300)
    # mrcnn.train_with_tfrecord(epochs=300, log_dir='./logs', tfrec_path='../data/voc_tfrec')
    mrcnn.train(epochs=300, log_dir='./logs', data_path='../data/voc2012_46_samples')

    # mrcnn.train_with_callback(epchos=101, log_dir='./logs')
    # mrcnn.test(model_path='./mrcnn-step-70000.h5', log_dir='./logs')
