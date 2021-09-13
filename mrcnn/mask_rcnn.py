import sys

sys.path.append("../../mask_rcnn_in_tf2_keras")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
from mrcnn.resnet import Resnet
from mrcnn.anchors_ops import get_anchors, AnchorsLayer
from mrcnn.bbox_ops import norm_boxes_graph, norm_boxes, denorm_boxes, unmold_mask, clip_boxes_graph
from mrcnn.layers import DetectionTargetLayer, ProposalLayer, FPNMaskLayer, PyramidROIAlignLayer
from mrcnn.layers import FPNClassifyLayer, AnchorTargetLayer, DetectionLayer, DetectionMaskLayer
from data.generate_voc_segment_data import VocSegmentDataGenerator
from mrcnn.losses import rpn_bbox_loss, rpn_class_loss
from mrcnn.losses import mrcnn_mask_loss, mrcnn_class_loss, mrcnn_bbox_loss
from data.visual_ops import draw_bounding_box, draw_instance
from data.generate_tfrecord_files import parse_voc_segment_tfrecord


class MaskRCNN:
    def __init__(self,
                 classes,
                 voc_data_path=None,
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
        self.voc_data_path = voc_data_path
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

            # loss
            # rpn_c_loss = tf.keras.layers.Lambda(lambda x: rpn_class_loss(*x),name="rpn_class_loss")(
            #     [rpn_target_match, rpn_class_logits])
            # rpn_b_loss = tf.keras.layers.Lambda(lambda x: rpn_bbox_loss(*x),name="rpn_bbox_loss")(
            #     [rpn_target_box, rpn_target_match, rpn_bbox_delta])
            # mrcnn_c_loss = tf.keras.layers.Lambda(lambda x: mrcnn_class_loss(*x), name="mrcnn_class_loss")(
            #     [mrcnn_target_class_ids, mrcnn_class_logits, rois])
            # mrcnn_b_loss = tf.keras.layers.Lambda(lambda x: mrcnn_bbox_loss(*x),name="mrcnn_bbox_loss")(
            #     [mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois])
            # mrcnn_m_loss = tf.keras.layers.Lambda(lambda x: mrcnn_mask_loss(*x), name="mrcnn_mask_loss")(
            #     [mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois])
            # total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_loss

            # Model
            inputs = [input_images, input_gt_boxes, gt_classes, gt_masks, all_anchors]
            # inputs = [input_images, input_gt_boxes, gt_classes, gt_masks, all_anchors, rpn_target_match, rpn_target_box]
            # outputs = [rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta,
            outputs = [rpn_class_logits, rpn_class, rpn_bbox_delta,
                       rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask]
            # outputs = [rpn_class_logits, rpn_class, rpn_bbox_delta,
            #            rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask,
            #            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
            #            rpn_c_loss, rpn_b_loss, mrcnn_c_loss,mrcnn_b_loss, mrcnn_m_loss]
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

    # def train_with_callback(self, epchos, log_dir):
    #     train_data = FitGenerator(
    #         voc_data_path=self.voc_data_path,
    #         classes=self.classes,
    #         is_training=True,
    #         batch_size=self.batch_size,
    #         im_size=self.image_shape[0],
    #         max_instance=40,
    #         use_mini_mask=self.use_mini_mask,
    #         mini_mask_shape=self.mini_mask_shape,
    #         anchor_scalse=self.scales,
    #         anchor_ratios=self.ratios,
    #         anchor_feature_strides=self.feature_strides,
    #         anchor_stride=self.anchor_stride,
    #         rpn_train_anchors_per_image=self.rpn_train_anchors_per_image,
    #         rpn_bbox_std_dev=self.rpn_bbox_std_dev
    #     )
    #     mrcnn = self.build_graph(is_training=True)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)
    #
    #     # add loss
    #     loss_names = [
    #         "rpn_class_loss",  "rpn_bbox_loss",
    #         "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
    #     for name in loss_names:
    #         layer = mrcnn.get_layer(name)
    #         # if layer.output in mrcnn.losses:
    #         #     continue
    #         mrcnn.add_loss(tf.reduce_mean(layer.output, keepdims=True))
    #
    #     # Add L2 Regularization
    #     # Skip gamma and beta weights of batch normalization layers.
    #     # reg_losses = [
    #     #     tf.keras.regularizers.l2(0.0001)(w) / tf.cast(tf.size(w), tf.float32)
    #     #     for w in mrcnn.trainable_weights
    #     #     if 'gamma' not in w.name and 'beta' not in w.name]
    #     # mrcnn.add_loss(tf.add_n(reg_losses))
    #
    #     # Compile
    #     mrcnn.compile(
    #         optimizer=optimizer,
    #         loss=[None] * len(mrcnn.outputs)
    #     )
    #
    #     # Add metrics for losses
    #     for name in loss_names:
    #         # if name in mrcnn.metrics_names:
    #         #     continue
    #         layer = mrcnn.get_layer(name)
    #         # mrcnn.add_metric(tf.reduce_mean(layer.output),name=name)
    #         mrcnn.metrics_names.append(name)
    #         mrcnn.metrics.append(tf.reduce_mean(layer.output, keepdims=True))
    #
    #     callbacks = [
    #         tf.keras.callbacks.TensorBoard(log_dir=log_dir,
    #                                     histogram_freq=0, write_graph=False, write_images=False)
    #         # tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
    #         #                                 verbose=0, save_weights_only=True),
    #     ]
    #
    #
    #     mrcnn.fit_generator(
    #         generator=train_data,
    #         initial_epoch=0,
    #         epochs=epchos,
    #         steps_per_epoch=700,
    #         callbacks=callbacks,
    #         validation_data=train_data,
    #         validation_steps=50,
    #         max_queue_size=100,
    #         workers=1,
    #         use_multiprocessing=True,
    #     )

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

    def train(self, epochs, log_dir):

        mrcnn = self.mrcnn
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        vsg_train = parse_voc_segment_tfrecord(
            is_training=True,
            tfrec_path='../../data/voc_tfrec',
            repeat=epochs,
            shuffle_buffer=800,
            batch=self.batch_size
        )
        anchors = get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              # scales=[192],
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              # feature_strides=[16],
                              anchor_stride=self.anchor_stride)
        all_anchors = np.stack([anchors, anchors], axis=0)

        summary_writer = tf.summary.create_file_writer(log_dir)

        # for epoch in range(epochs):
        #     for batch in range(vsg_train.total_batch_size):
        #         imgs, masks, gt_boxes, labels = vsg_train.next_batch()
        step = -1
        for imgs, masks, gt_boxes, labels, rpn_target_match, rpn_target_box in vsg_train:
            if np.sum(gt_boxes) <= 0.:
                print(step, " gt_boxes: ", gt_boxes)
                continue

            if step % 2000 == 0:
                mrcnn.save_weights("./mrcnn-step-{}.h5".format(step))
            step += 1

            with tf.GradientTape() as tape:
                # 模型输出
                # rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask, mrcnn_class_logits, \
                mrcnn_class, mrcnn_bbox, mrcnn_mask = \
                    mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)
                # mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)


                rpn_c_loss = rpn_class_loss(rpn_target_match, rpn_class_logits)
                rpn_b_loss = rpn_bbox_loss(rpn_target_box, rpn_target_match, rpn_bbox_delta)
                mrcnn_c_loss = mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_class_logits, rois)
                mrcnn_b_loss = mrcnn_bbox_loss(mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois)
                mrcnn_m_bc_loss = mrcnn_mask_loss(mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois)
                total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_bc_loss

                # 梯度更新
                grad = tape.gradient(total_loss, mrcnn.trainable_variables)
                optimizer.apply_gradients(zip(grad, mrcnn.trainable_variables))

                # tensorboard
                with summary_writer.as_default():
                    tf.summary.scalar('loss/rpn_class_loss', rpn_c_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)
                    tf.summary.scalar('loss/rpn_bbox_loss', rpn_b_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)
                    tf.summary.scalar('loss/mrcnn_class_loss', mrcnn_c_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)
                    tf.summary.scalar('loss/mrcnn_bbox_loss', mrcnn_b_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)
                    tf.summary.scalar('loss/mrcnn_mask_binary_crossentropy_loss', mrcnn_m_bc_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)
                    tf.summary.scalar('loss/total_loss', total_loss,
                                      # step=epoch * vsg_train.total_batch_size + batch)
                                      step=step)

                # [b, num_detections, (y1, x1, y2, x2, class_id, score)], [b, num_detections, h, w, num_classes]
                detections, pred_masks = DetectionMaskLayer(
                    batch_size=self.batch_size,
                    bbox_std_dev=self.bbox_std_dev,
                    detection_max_instances=self.detection_max_instances,
                    detection_nms_thres=self.detection_nms_thres,
                    detection_min_confidence=self.detection_min_confidence
                )(rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, np.array([0, 0, 1, 1], np.float32))

                for i in range(self.batch_size):
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

                    # non_zeros = tf.cast(tf.reduce_sum(tf.abs(mrcnn_target_bbox[i]), axis=1), tf.bool)
                    # target_masks = tf.boolean_mask(mrcnn_target_mask[i], non_zeros)
                    # target_classes = tf.boolean_mask(mrcnn_target_class_ids[i], non_zeros)
                    # target_rois = tf.boolean_mask(rois[i], non_zeros)
                    # target_rois = denorm_boxes(target_rois, self.image_shape[0:2])
                    # # print(tf.shape(target_rois),tf.shape(target_masks),tf.shape(target_classes))
                    #
                    # for j in range(len(target_classes)):
                    #     l = target_classes[j]
                    #     class_name = self.classes[target_classes[j]]
                    #     ymin, xmin, ymax, xmax = target_rois[j]
                    #     gt_img = draw_bounding_box(gt_img, class_name, l, xmin, ymin, xmax, ymax)
                    #     full_mask = unmold_mask(np.array(target_masks[j]), target_rois[j], self.image_shape)
                    #     gt_img = draw_instance(gt_img, full_mask)

                    concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                    summ_imgs = tf.expand_dims(concat_imgs, 0)
                    summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                    with summary_writer.as_default():
                        tf.summary.image("imgs/gt,pred,epoch{}".format(step // 1500), summ_imgs, step=step)

    def test(self, model_path, log_dir):

        self.load_weights(model_path, by_name=True)
        summary_writer = tf.summary.create_file_writer(log_dir)
        vsg_train = parse_voc_segment_tfrecord(
            is_training=True,
            tfrec_path='../../data/voc_tfrec',
            repeat=1,
            shuffle_buffer=800,
            batch=self.batch_size
        )

        anchors = get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              anchor_stride=self.anchor_stride)
        all_anchors = np.stack([anchors, anchors], axis=0)

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


if __name__ == "__main__":
    mrcnn = MaskRCNN(classes=['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                     voc_data_path='../../data/VOCdevkit/VOC2012/',
                     is_training=True
                     )
    # mrcnn = MaskRCNN(classes=['_background_', 'aeroplane'],
    #                  voc_data_path='../../data/VOCdevkit/VOC2012/'
    #                  )
    # model = mrcnn.build_graph(is_training=False)
    # model.summary(line_length=300)
    mrcnn.train(epochs=300, log_dir='./logs')

    # mrcnn.train_with_callback(epchos=101, log_dir='./logs')
    # mrcnn.test(model_path='./mrcnn-step-70000.h5', log_dir='./logs')
