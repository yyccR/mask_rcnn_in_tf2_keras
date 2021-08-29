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

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)(y_true=anchor_class, y_pred=rpn_class_logits)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss

def rpn_bbox_loss(target_bbox, rpn_match, predict_bbox):
    """ 计算rpn预测的box损失

    :param target_bbox: [batch, nums, 4]
    :param rpn_match: [batch, nums]
    :param predict_bbox: [batch, nums, 4]
    :return:
    """
    # 只要label=1的那些box
    indices = tf.where(rpn_match == 1)

    # Pick bbox deltas that contribute to the loss
    target_bbox = tf.gather_nd(target_bbox, indices)
    predict_bbox = tf.gather_nd(predict_bbox, indices)
    loss = smooth_l1_loss(target_bbox, predict_bbox)

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
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(rois), axis=2), tf.bool)
    target_class_ids = tf.cast(tf.boolean_mask(target_class_ids, non_zeros),dtype=tf.int64)
    pred_class_logits = tf.cast(tf.boolean_mask(pred_class_logits, non_zeros),dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox, rois):
    """ mrcnn层预测box损失

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    positive_roi_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0],dtype=tf.int32)
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), dtype=tf.int32)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
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
    loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                    # tf.keras.losses.BinaryCrossentropy()(y_true=y_true, y_pred=y_pred),
                    tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss


