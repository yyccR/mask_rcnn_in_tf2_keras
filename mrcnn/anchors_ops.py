import math
import numpy as np
import tensorflow as tf
from mrcnn.bbox_ops import norm_boxes


def compute_backbone_shapes(image_shape, backbone_strides=[4, 8, 16, 32, 64]):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """

    # Currently supports ResNet only
    return np.array([[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
                     for stride in backbone_strides])


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    :param scales [32, 64, 128, 256, 512]
    :param ratios [0.5, 1, 2]
    :param feature_shapes [image_height / [4, 8, 16, 32, 64], image_width / [4, 8, 16, 32, 64]]
    :param feature_strides [4, 8, 16, 32, 64]
    :param anchor_stride 1
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # def log2_graph(x):
    #     """Implementation of Log2. TF doesn't have a native implementation."""
    #     return tf.math.log(x) / tf.math.log(2.0)

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        # a = generate_anchors(scales[i], ratios, feature_shapes[i],
        #                                 feature_strides[i], anchor_stride)
        # na = norm_boxes(a, (320.,320.))
        # for j in na:
        #     y1, x1, y2, x2 = j
        #     h = y2 - y1
        #     w = x2 - x1
        #     roi_level = log2_graph(tf.cast(tf.sqrt(h * w), dtype=tf.float32) / (224.0 / tf.sqrt(320. * 320.)))
        #     roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        #     print(scales[i], roi_level)
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def get_anchors(image_shape, scales, ratios, feature_strides, anchor_stride):
    """Returns anchor pyramid for the given image size."""
    backbone_shapes = compute_backbone_shapes(image_shape=image_shape, backbone_strides=feature_strides)
    # print(backbone_shapes)
    # Generate Anchors
    a = generate_pyramid_anchors(
        scales=scales,
        ratios=ratios,
        feature_shapes=backbone_shapes,
        feature_strides=feature_strides,
        anchor_stride=anchor_stride)

    # Normalize coordinates
    a = norm_boxes(a, image_shape[:2])
    return a


class AnchorsLayer(tf.keras.layers.Layer):
    def __init__(self,
                 image_shape,
                 scales=[32, 64, 128, 256, 512],
                 ratios=[0.5, 1, 2],
                 feature_strides=[4, 8, 16, 32, 64],
                 anchor_stride=1,
                 ):
        super(AnchorsLayer, self).__init__()
        self.image_shape = image_shape
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
        self.anchor_stride = anchor_stride

    def compute_backbone_shapes(self, image_shape, backbone_strides=[4, 8, 16, 32, 64]):
        """Computes the width and height of each stage of the backbone network.
        Returns:
            [N, (height, width)]. Where N is the number of stages
        """

        # Currently supports ResNet only
        return np.array([[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
                         for stride in backbone_strides])

    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides, anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.
        :param scales [32, 64, 128, 256, 512]
        :param ratios [0.5, 1, 2]
        :param feature_shapes [image_height / [4, 8, 16, 32, 64], image_width / [4, 8, 16, 32, 64]]
        :param feature_strides [4, 8, 16, 32, 64]
        :param anchor_stride 1
        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # def log2_graph(x):
        #     """Implementation of Log2. TF doesn't have a native implementation."""
        #     return tf.math.log(x) / tf.math.log(2.0)

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            # a = generate_anchors(scales[i], ratios, feature_shapes[i],
            #                                 feature_strides[i], anchor_stride)
            # na = norm_boxes(a, (320.,320.))
            # for j in na:
            #     y1, x1, y2, x2 = j
            #     h = y2 - y1
            #     w = x2 - x1
            #     roi_level = log2_graph(tf.cast(tf.sqrt(h * w), dtype=tf.float32) / (224.0 / tf.sqrt(320. * 320.)))
            #     roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
            #     print(scales[i], roi_level)
            anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                            feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    def get_anchors(self, image_shape, scales, ratios, feature_strides, anchor_stride):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(image_shape=image_shape, backbone_strides=feature_strides)
        # print(backbone_shapes)
        # Generate Anchors
        a = generate_pyramid_anchors(
            scales=scales,
            ratios=ratios,
            feature_shapes=backbone_shapes,
            feature_strides=feature_strides,
            anchor_stride=anchor_stride)

        # Normalize coordinates
        a = norm_boxes(a, image_shape[:2])
        return a

    def call(self, inputs):
        na = self.get_anchors(image_shape=self.image_shape,
                              scales=self.scales,
                              ratios=self.ratios,
                              feature_strides=self.feature_strides,
                              anchor_stride=self.anchor_stride)
        na = tf.convert_to_tensor(na)
        return na


def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


