import sys

sys.path.append("../../mask_rcnn_in_tf2_keras")

import os
import tensorflow as tf
import numpy as np
from mrcnn.mask_rcnn import MaskRCNN
from mrcnn.anchors_ops import get_anchors
from mrcnn.layers import build_rpn_targets, DetectionMaskLayer
from mrcnn.bbox_ops import unmold_mask
from mrcnn.losses import rpn_bbox_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss
from data.generate_voc_segment_data import VocSegmentDataGenerator
from data.generate_coco_data import CoCoDataGenrator
from data.visual_ops import draw_bounding_box, draw_instance

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    epochs = 300
    batch_size = 2
    image_shape = [320, 320, 3]
    use_mini_mask = True
    mini_mask_shape = (56, 56)
    detection_max_instances = 100
    scales = [32, 64, 128, 256, 512]
    ratios = [0.5, 1, 2]
    feature_strides = [4, 8, 16, 32, 64]
    anchor_stride = 1
    pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])
    # 每个类别最大训练数据数
    data_max_size_per_class = 100

    # voc data class
    classes = ['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    voc_data_path = "../data/voc2012_46_samples"
    # voc格式数据
    data_train = VocSegmentDataGenerator(
        classes=classes,
        voc_data_path=voc_data_path,
        batch_size=batch_size,
        is_training=True,
        im_size=image_shape[0],
        data_max_size_per_class=data_max_size_per_class,
        max_instance=detection_max_instances,
        image_mean=pixel_mean,
        use_mini_mask=use_mini_mask,
        mini_mask_shape=mini_mask_shape
    )

    # tensorboard 日志目录
    log_dir = "./logs"
    summary_writer = tf.summary.create_file_writer(log_dir)
    mrcnn = MaskRCNN(classes=classes,
                     is_training=True,
                     batch_size=batch_size,
                     image_shape=image_shape,
                     use_mini_mask=use_mini_mask,
                     mini_mask_shape=mini_mask_shape,
                     detection_max_instances=detection_max_instances,
                     pixel_mean=pixel_mean,
                     scales=scales,
                     ratios=ratios
                     )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    anchors = get_anchors(image_shape=image_shape,
                          scales=scales,
                          ratios=ratios,
                          feature_strides=feature_strides,
                          anchor_stride=anchor_stride)
    # all_anchors = np.stack([anchors, anchors], axis=0)
    all_anchors = np.tile([anchors], reps=[batch_size, 1, 1])

    for epoch in range(epochs):
        for batch in range(data_train.total_batch_size):
            imgs, masks, gt_boxes, labels = data_train.next_batch()
            rpn_target_match, rpn_target_box = build_rpn_targets(
                anchors=anchors,
                gt_boxes=gt_boxes,
                image_shape=image_shape[0:2],
                batch_size=batch_size,
                rpn_train_anchors_per_image=mrcnn.rpn_train_anchors_per_image,
                rpn_bbox_std_dev=mrcnn.rpn_bbox_std_dev)

            print(np.shape(imgs))
            print(np.shape(masks))
            print(np.shape(gt_boxes))
            print("-------{}-{}--------".format(epoch, batch))

            if np.sum(gt_boxes) <= 0.:
                print(batch, " gt_boxes: ", gt_boxes)
                continue

            if epoch % 20 == 0 and epoch != 0:
                mrcnn.model.save_weights("./mrcnn-epoch-{}.h5".format(epoch))

            with tf.GradientTape() as tape:
                # 模型输出
                # rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
                mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask, mrcnn_class_logits, \
                mrcnn_class, mrcnn_bbox, mrcnn_mask = \
                    mrcnn.model([imgs, gt_boxes, labels, masks, all_anchors], training=True)
                # mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)

                # 计算损失
                rpn_c_loss = rpn_class_loss(rpn_target_match, rpn_class_logits)
                rpn_b_loss = rpn_bbox_loss(rpn_target_box, rpn_target_match, rpn_bbox_delta)
                mrcnn_c_loss = mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_class_logits, rois)
                mrcnn_b_loss = mrcnn_bbox_loss(mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois)
                mrcnn_m_bc_loss = mrcnn_mask_loss(mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois)
                total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_bc_loss

                # 梯度更新
                grad = tape.gradient(total_loss, mrcnn.model.trainable_variables)
                optimizer.apply_gradients(zip(grad, mrcnn.model.trainable_variables))

                # tensorboard 损失曲线
                with summary_writer.as_default():
                    tf.summary.scalar('loss/rpn_class_loss', rpn_c_loss,
                                      step=epoch * data_train.total_batch_size + batch)
                    tf.summary.scalar('loss/rpn_bbox_loss', rpn_b_loss,
                                      step=epoch * data_train.total_batch_size + batch)
                    tf.summary.scalar('loss/mrcnn_class_loss', mrcnn_c_loss,
                                      step=epoch * data_train.total_batch_size + batch)
                    tf.summary.scalar('loss/mrcnn_bbox_loss', mrcnn_b_loss,
                                      step=epoch * data_train.total_batch_size + batch)
                    tf.summary.scalar('loss/mrcnn_mask_binary_crossentropy_loss', mrcnn_m_bc_loss,
                                      step=epoch * data_train.total_batch_size + batch)
                    tf.summary.scalar('loss/total_loss', total_loss,
                                      step=epoch * data_train.total_batch_size + batch)

                # 非极大抑制与其他条件过滤
                # [b, num_detections, (y1, x1, y2, x2, class_id, score)], [b, num_detections, h, w, num_classes]
                detections, pred_masks = DetectionMaskLayer(
                    batch_size=batch_size,
                    bbox_std_dev=mrcnn.bbox_std_dev,
                    detection_max_instances=detection_max_instances,
                    detection_nms_thres=mrcnn.detection_nms_thres,
                    detection_min_confidence=mrcnn.detection_min_confidence
                )(rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, np.array([0, 0, 1, 1], np.float32))

                for i in range(batch_size):
                    # 将数据处理成原图大小
                    boxes, class_ids, scores, full_masks = mrcnn.unmold_detections(
                        detections=detections[i],
                        mrcnn_mask=pred_masks[i],
                        original_image_shape=image_shape)

                    # 预测结果
                    pred_img = imgs[i].copy() + pixel_mean
                    for j in range(np.shape(class_ids)[0]):
                        score = scores[j]
                        if score > 0.1:
                            class_name = classes[class_ids[j]]
                            ymin, xmin, ymax, xmax = boxes[j]
                            pred_mask_j = full_masks[:, :, j]
                            pred_img = draw_instance(pred_img, pred_mask_j)
                            pred_img = draw_bounding_box(pred_img, class_name, score, xmin, ymin, xmax, ymax)

                    # ground true
                    gt_img = imgs[i].copy() + pixel_mean
                    active_num = len(np.where(labels[i])[0])
                    for j in range(active_num):
                        l = labels[i][j]
                        class_name = classes[l]
                        ymin, xmin, ymax, xmax = gt_boxes[i][j]
                        gt_mask_j = unmold_mask(np.array(masks[i][:, :, j], dtype=np.float32), gt_boxes[i][j],
                                                image_shape)
                        gt_img = draw_bounding_box(gt_img, class_name, l, xmin, ymin, xmax, ymax)
                        gt_img = draw_instance(gt_img, gt_mask_j)

                    concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                    summ_imgs = tf.expand_dims(concat_imgs, 0)
                    summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                    with summary_writer.as_default():
                        tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs, step=batch)

    mrcnn.model.save_weights("./mrcnn-epoch-{}.h5".format(epochs))

if __name__ == "__main__":
    main()
