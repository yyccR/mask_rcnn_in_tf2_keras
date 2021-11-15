import sys
import os
import tensorflow as tf
import numpy as np


def tensor_feature(value):
    """ tensor序列化成feature
    :param value:
    :return:
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_voc_segment_tfrecord(is_training=True, tfrec_path='./voc_tfrec/', voc_data_path='./voc2012_46_samples'):
    from data.generate_voc_segment_data import VocSegmentDataGenerator
    from mrcnn.layers import build_rpn_targets
    from mrcnn.anchors_ops import get_anchors

    if not os.path.isdir(tfrec_path):
        os.mkdir(tfrec_path)

    voc_seg = VocSegmentDataGenerator(
        voc_data_path=voc_data_path,
        batch_size=1,
        class_balance=True,
        is_training=is_training,
        im_size=640,
        data_max_size_per_class=2
    )

    if is_training:
        tfrec_file = os.path.join(tfrec_path, "voc_train_seg.tfrec")
    else:
        tfrec_file = os.path.join(tfrec_path, "voc_test_seg.tfrec")
    tfrec_writer = tf.io.TFRecordWriter(tfrec_file)

    anchors = get_anchors(image_shape=[640, 640],
                          scales=[32, 64, 128, 256, 512],
                          ratios=[0.5, 1, 2],
                          feature_strides=[4, 8, 16, 32, 64],
                          anchor_stride=1)

    for i in range(voc_seg.total_batch_size):
        print("current {} total {}".format(i, voc_seg.total_batch_size))
        # for batch in range(vsg_train.total_batch_size):
        imgs, masks, gt_boxes, labels = voc_seg.next_batch()
        # indices = voc_seg.file_indices[i * 1: (i + 1) * 1]
        # cur_img_files = [voc_seg.img_files[k] for k in indices]
        # cur_cls_files = [voc_seg.cls_mask_files[k] for k in indices]
        # cur_obj_files = [voc_seg.obj_mask_files[k] for k in indices]
        # imgs, masks, gt_boxes, labels = voc_seg._data_generation(img_files=cur_img_files,
        #                                                          cls_files=cur_cls_files,
        #                                                          obj_files=cur_obj_files)
        rpn_target_match, rpn_target_box = build_rpn_targets(anchors=anchors,
                                                             gt_boxes=gt_boxes,
                                                             image_shape=[640, 640],
                                                             batch_size=1,
                                                             rpn_train_anchors_per_image=256,
                                                             rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]))
        feature = {
            "image": tensor_feature(imgs),
            "masks": tensor_feature(masks),
            "gt_boxes": tensor_feature(gt_boxes),
            "labels": tensor_feature(labels),
            "rpn_target_match": tensor_feature(rpn_target_match),
            "rpn_target_box": tensor_feature(rpn_target_box)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tfrec_writer.write(example.SerializeToString())
    tfrec_writer.close()


def parse_single_example(single_record):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "masks": tf.io.FixedLenFeature([], tf.string),
        "gt_boxes": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
        "rpn_target_match": tf.io.FixedLenFeature([], tf.string),
        "rpn_target_box": tf.io.FixedLenFeature([], tf.string)
    }
    feature = tf.io.parse_single_example(single_record, feature_description)

    image = tf.io.parse_tensor(feature['image'], tf.float64)[0]
    masks = tf.io.parse_tensor(feature['masks'], tf.int8)[0]
    gt_boxes = tf.io.parse_tensor(feature['gt_boxes'], tf.float32)[0]
    labels = tf.io.parse_tensor(feature['labels'], tf.int8)[0]
    rpn_target_match = tf.io.parse_tensor(feature['rpn_target_match'], tf.float32)[0]
    rpn_target_box = tf.io.parse_tensor(feature['rpn_target_box'], tf.float32)[0]

    return image, masks, gt_boxes, labels,rpn_target_match,rpn_target_box


def parse_voc_segment_tfrecord(is_training=True, tfrec_path='./voc_tfrec', repeat=1, shuffle_buffer=1000, batch=2):
    if is_training:
        tfrec_file = os.path.join(tfrec_path, "voc_train_seg.tfrec")
    else:
        tfrec_file = os.path.join(tfrec_path, "voc_test_seg.tfrec")

    voc_tfrec_dataset = tf.data.TFRecordDataset(tfrec_file,num_parallel_reads=2)
    parse_data = voc_tfrec_dataset\
        .repeat(repeat)\
        .shuffle(shuffle_buffer)\
        .map(parse_single_example)\
        .batch(batch, drop_remainder=True)\
        .prefetch(10)

    return parse_data


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import time

    generate_voc_segment_tfrecord()
    # parse_data = parse_voc_segment_tfrecord()
    # i = 0
    # for image, masks, gt_boxes, labels,rpn_target_match,rpn_target_box in parse_data:
    #     print(image.shape)
    #     print(masks.shape)
    #     print(gt_boxes.shape)
    #     print(labels.shape)
    #     print(rpn_target_match.shape)
    #     print(rpn_target_box.shape)
    #     i += 1
    #     print(i)

    # data = parse_data.get_next_as_optional()
    # print(tf.io.parse_tensor(data['gt_boxes'].numpy(), tf.float32).shape)
    # print(tf.io.parse_tensor(data['masks'].numpy(), tf.int8).shape)
    # print(tf.io.parse_tensor(data['image'].numpy(), tf.float64).shape)
    # x = tf.random.normal([2, 2, 2, 2])
    # xs = tf.io.serialize_tensor(x)
    # xf = tf.train.Feature(bytes_list=tf.train.BytesList(value=[xs.numpy()]))
    # f = {"xf": xf}
    # example = tf.train.Example(features=tf.train.Features(feature=f))
    # with tf.io.TFRecordWriter('./test.tfrec') as writer:
    #     writer.write(example.SerializeToString())
    #
    # parse = lambda x: tf.io.parse_single_example(x, {"xf": tf.io.FixedLenFeature([], tf.string)})
    # data = tf.data.TFRecordDataset('./test.tfrec')
    # parse_data = data.map(parse)
    # for features in parse_data.take(1):
    #     print(tf.io.parse_tensor(features['xf'].numpy(), tf.float32))

