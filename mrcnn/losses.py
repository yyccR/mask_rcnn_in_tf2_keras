import tensorflow as tf


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.keras.backend.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.keras.backend.less(diff, 1.0), dtype=tf.float32)
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss(rpn_match, rpn_class_logits):
    """ rpn网络的0、1分类损失计算

    rpn_match: [batch, anchors]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    # rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(rpn_match == 1, tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(rpn_match != 0)
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    #
    # rpn_match = tf.reshape(rpn_match, (-1, ))
    # rpn_class_logits = tf.reshape(rpn_class_logits, (-1, 2))
    # rpn_fg = tf.where(rpn_match == 1)[:,0]
    # rpn_nums = tf.cast(tf.shape(rpn_fg)[0], dtype=tf.float32) * 0.5
    # rpn_nums = tf.cast(tf.math.floor(rpn_nums), dtype=tf.int32)
    # rpn_fg_label = tf.gather(rpn_match, rpn_fg)
    # rpn_bg = tf.random.shuffle(tf.where(rpn_match == -1)[:,0])[:rpn_nums]
    # rpn_bg_label = tf.gather(rpn_match, rpn_bg) * 0.
    # rpn_idx = tf.concat([rpn_fg, rpn_bg], axis=0)
    # anchor_class = tf.concat([rpn_fg_label, rpn_bg_label], axis=0)
    # rpn_class_logits = tf.gather(rpn_class_logits, rpn_idx)

    # # bg_num = tf.shape(tf.where(rpn_match == -1)[:,0])[0]
    # # fg_num = tf.shape(tf.where(rpn_match == 1)[:,0])[0]
    # # print(fg_num, bg_num)
    # # print(rpn_fg)
    # # print(rpn_bg)
    # print(anchor_class)
    # print(tf.argmax(rpn_class_logits, axis=1))
    # tmp = tf.concat([rpn_class_logits, tf.expand_dims(tf.cast(anchor_class,tf.float32),1)], axis=1)
    # print(tmp)
    # Cross entropy loss
    # loss = tf.keras.losses.sparse_categorical_crossentropy(target=anchor_class,
    #                                          output=rpn_class_logits,
    #                                          from_logits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)(y_true=anchor_class, y_pred=rpn_class_logits)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss

from models.mask_rcnn.anchors_ops import get_anchors
def rpn_bbox_loss(target_bbox, rpn_match, predict_bbox):
    """ 计算rpn预测的box损失

    :param target_bbox: [batch, nums, 4]
    :param rpn_match: [batch, nums]
    :param predict_bbox: [batch, nums, 4]
    :return:
    """
    # anchors = get_anchors(image_shape=[640,640,3],
    #                       scales=[32, 64, 128, 256, 512],
    #                       # scales=[256],
    #                       ratios=[0.5, 1, 2],
    #                       feature_strides=[4, 8, 16, 32, 64],
    #                       # feature_strides=[16],
    #                       anchor_stride=1)
    # indices = tf.where(rpn_match[0] == 1)[: 0]
    # print(indices)
    # print(tf.gather(anchors, indices))


    # rpn_match = K.squeeze(rpn_match, -1)
    # 只要label=1的那些box
    indices = tf.where(rpn_match == 1)
    # print(anchors)
    # print(indices)
    # print(tf.gather(anchors, indices[:, 1]))
    # print(indices)
    # print(target_bbox)
    # print(predict_bbox)
    # print("------------------------")

    # Pick bbox deltas that contribute to the loss
    target_bbox = tf.gather_nd(target_bbox, indices)
    # print("------------target box------------")
    # print(target_bbox)
    predict_bbox = tf.gather_nd(predict_bbox, indices)
    # print("------------pred box--------------")
    # print(predict_bbox)
    # print(target_bbox)
    # print(predict_bbox)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    # batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    # target_bbox = batch_pack_graph(target_bbox, batch_counts,
    #                                config.IMAGES_PER_GPU)
    # print(target_bbox)
    # print(predict_bbox)
    loss = smooth_l1_loss(target_bbox, predict_bbox)

    # loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss(target_class_ids, pred_class_logits, rois):
    """ mrcnn层预测类别损失

    :param target_class_ids: [batch, nums]
    :param pred_class_logits: [batch, nums, num_classes]
    :param rois: [batch, nums, [dy, dx, dh ,dw]]
    :return:
    """
    # 排除那些padding=0的
    # print(target_class_ids)
    # print(pred_class_logits)

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(rois), axis=2), tf.bool)
    target_class_ids = tf.cast(tf.boolean_mask(target_class_ids, non_zeros),dtype=tf.int64)
    pred_class_logits = tf.cast(tf.boolean_mask(pred_class_logits, non_zeros),dtype=tf.float32)
    # pred_class_logits = tf.argmax(pred_class_logits, axis=2)
    # active_class = tf.where(target_class_ids > 0)
    # target_class_ids = tf.cast(tf.gather_nd(target_class_ids, active_class), dtype=tf.int32)
    # pred_class_logits = tf.gather_nd(pred_class_logits, active_class)
    # Find predictions of classes that are not in the dataset.
    # pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    # pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # pred_shape = tf.shape(pred_class_logits)
    # target_class_ids = tf.reshape(target_class_ids, (-1, ))
    # pred_class_logits = tf.reshape(pred_class_logits, (-1, pred_shape[2]))
    # fg_ids = tf.where(target_class_ids > 0)[:,0]
    # bg_nums = tf.cast(tf.shape(fg_ids)[0], dtype=tf.float32) * 0.5
    # bg_nums = tf.cast(tf.math.floor(bg_nums), dtype=tf.int32)
    # fg_label = tf.gather(target_class_ids, fg_ids)
    # bg_ids = tf.random.shuffle(tf.where(target_class_ids == 0)[:,0])[:bg_nums]
    # bg_label = tf.gather(target_class_ids, bg_ids)
    # fg_bg_ids = tf.concat([fg_ids, bg_ids], axis=0)
    # target_class_ids = tf.concat([fg_label, bg_label], axis=0)
    # pred_class_logits = tf.gather(pred_class_logits, fg_bg_ids)

    # Loss
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=target_class_ids, logits=pred_class_logits)
    # print(fg_bg_labels)
    # print("----------------------")
    # print(target_class_ids)
    # print(tf.argmax(pred_class_logits,axis=1))
    # print(pred_class_logits)
    # print(pred_class_logits)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
    #     y_true=target_class_ids, y_pred=pred_class_logits)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    # loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    # loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    # loss = tf.reduce_sum(loss)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox, rois):
    """ mrcnn层预测box损失

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    # 排除rois那些padding=0的
    # print(rois)
    # print(target_bbox[0])
    # print(pred_bbox[0])
    # non_zeros = tf.cast(tf.reduce_sum(tf.abs(rois), axis=2), tf.bool)
    # # # print(non_zeros)
    # # rois = tf.boolean_mask(rois, non_zeros)
    # target_bbox = tf.boolean_mask(target_bbox, non_zeros)
    # target_class_ids = tf.boolean_mask(target_class_ids, non_zeros)
    # pred_bbox = tf.boolean_mask(pred_bbox, non_zeros)
    # print(tf.shape(target_bbox))
    # print(tf.shape(target_class_ids))
    # print(tf.shape(pred_bbox))
    # print("1111111111111111111111111111")
    # print(non_zeros)
    # print(rois)
    # print(target_bbox)
    # print(pred_bbox)

    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))
    # pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[1], 4))
    # print(pred_bbox)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0],dtype=tf.int32)
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), dtype=tf.int32)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    # print("22222222222222222222222222")
    # print(target_bbox)
    # print(pred_bbox)
    # Smooth-L1 Loss
    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                                   tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss


def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks, rois):
    """ mrcnn预测mask损失

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, num_rois, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    mask_shape = tf.shape(target_masks)
    pred_shape = tf.shape(pred_masks)

    # non_zeros = tf.cast(tf.reduce_sum(tf.abs(rois), axis=2), tf.bool)
    # # # print(non_zeros)
    # # rois = tf.boolean_mask(rois, non_zeros)
    # target_masks = tf.boolean_mask(target_masks, non_zeros)
    # target_class_ids = tf.boolean_mask(target_class_ids, non_zeros)
    # pred_masks = tf.boolean_mask(pred_masks, non_zeros)

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    # mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    # pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    bc_loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                    # tf.keras.losses.BinaryCrossentropy()(y_true=y_true, y_pred=y_pred),
                    tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    bc_loss = tf.keras.backend.mean(bc_loss)

    # mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2])
    # mae_loss = tf.reduce_mean(mae_loss)

    # return bc_loss, mae_loss
    return bc_loss


def l2_regularize_loss(model, weight_decay=0.0001):
    loss = tf.constant(0.0, dtype=tf.float32)
    for w in model.trainable_weights:
        if 'gamma' not in w.name and 'beta' not in w.name:
            loss += tf.keras.regularizers.l2(weight_decay)(w) / tf.cast(tf.size(w), tf.float32)
    return loss
