import numpy as np
import tensorflow as tf
from models.mask_rcnn.bbox_ops import clip_boxes_graph, apply_box_deltas_graph, norm_boxes_graph
from models.mask_rcnn.bbox_ops import trim_zeros_graph, overlaps_graph, box_refinement_graph


class ProposalLayer(tf.keras.layers.Layer):
    """ rpn数据过滤层, 对score排序和采样,再将预测的box delta映射到对应的anchor上 """

    def __init__(self, proposal_count, nms_threshold, rpn_bbox_std_dev, rpn_nms_limit=6000, batch_size=2):
        super(ProposalLayer, self).__init__()
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.rpn_nms_limit = rpn_nms_limit
        self.batch_size = batch_size

    def nms(self, boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(boxes, indices)
        # Pad if needed
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals

    def call(self, rpn_scores, rpn_bbox_delta, anchors):
        """
        :param rpn_scores: [batch, num_anchors, (bg prob, fg prob)]
        :param rpn_bbox_delta: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        :param anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
        :return:
        """
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = rpn_scores[:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = rpn_bbox_delta * np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.rpn_nms_limit, tf.shape(anchors)[0])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        target_scores = []
        target_deltas = []
        target_anchors = []
        for i in range(self.batch_size):
            target_scores.append(tf.gather(scores[i], ix[i]))
            target_deltas.append(tf.gather(deltas[i], ix[i]))
            target_anchors.append(tf.gather(anchors, ix[i]))

        target_scores = tf.stack(target_scores, axis=0)
        target_deltas = tf.stack(target_deltas, axis=0)
        target_anchors = tf.stack(target_anchors, axis=0)

        # [batch, N, (x1, y1, x2, y2)]
        inv_transform_boxes = []
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        for i in range(self.batch_size):
            cur_boxes = apply_box_deltas_graph(target_anchors[i], target_deltas[i])
            clip_boxes = clip_boxes_graph(cur_boxes, window)
            inv_transform_boxes.append(clip_boxes)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.
        # Non-max suppression
        proposals = []
        for i in range(self.batch_size):
            nms_boxes = self.nms(inv_transform_boxes[i], target_scores[i])
            proposals.append(nms_boxes)

        proposals = tf.stack(proposals, axis=0)
        return proposals


def random_disable_labels(labels_input, inds, disable_nums):
    shuffle_fg_inds = tf.random.shuffle(inds)
    disable_inds = shuffle_fg_inds[:disable_nums]
    disable_inds_expand_dim = tf.expand_dims(disable_inds, axis=1)
    zeros = tf.zeros_like(disable_inds, dtype=tf.float32)
    return tf.tensor_scatter_nd_update(labels_input, disable_inds_expand_dim, zeros)


def build_rpn_targets(anchors, gt_boxes, image_shape, batch_size, rpn_train_anchors_per_image, rpn_bbox_std_dev):
    """ 生成rpn的目标数据, 做损失计算用 """
    batch_rpn_match = []
    batch_rpn_bbox = []
    for i in range(batch_size):
        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = tf.zeros([tf.shape(anchors)[0]], dtype=tf.float32)
        cur_gt_boxes = gt_boxes[i]
        cur_gt_boxes = norm_boxes_graph(cur_gt_boxes, image_shape)
        # 去掉那些padding的数据
        cur_gt_boxes, non_zeros = trim_zeros_graph(cur_gt_boxes, name="trim_gt_boxes")
        overlaps = overlaps_graph(anchors, cur_gt_boxes)

        pos_values = tf.ones_like(rpn_match)
        neg_values = tf.ones_like(rpn_match) * -1

        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        anchor_iou_max = tf.keras.backend.max(overlaps, axis=1)

        # iou 小于 0.3的为背景样本
        rpn_match = tf.where(anchor_iou_max < 0.3, neg_values, rpn_match)

        # 跟gt_box iou最大的都标上前景标签
        gt_iou_argmax = tf.cast(tf.argmax(overlaps, axis=0), dtype=tf.int32)
        pos_update = tf.ones_like(gt_iou_argmax, dtype=tf.float32)
        gt_iou_argmax_expand = tf.expand_dims(gt_iou_argmax, axis=1)
        rpn_match = tf.tensor_scatter_nd_update(rpn_match, gt_iou_argmax_expand, pos_update)

        # iou 大于 0.7的为前景样本
        rpn_match = tf.where(anchor_iou_max > 0.5, pos_values, rpn_match)

        # 采样保证正负样本在一定数量范围内
        pos_ids = tf.where(rpn_match == 1)[:, 0]
        pos_extra = tf.shape(pos_ids)[0] - (rpn_train_anchors_per_image // 2)
        fg_flag = tf.cast(pos_extra > 0, dtype=tf.float32)
        rpn_match = fg_flag * random_disable_labels(rpn_match, pos_ids, pos_extra) + \
                    (1.0 - fg_flag) * rpn_match

        neg_ids = tf.where(rpn_match == -1)[:, 0]
        neg_extra = tf.shape(neg_ids)[0] - \
                    (rpn_train_anchors_per_image - tf.shape(tf.where(rpn_match == 1)[:, 0])[0])
        bg_flag = tf.cast(neg_extra > 0, dtype=tf.float32)
        rpn_match = bg_flag * random_disable_labels(rpn_match, neg_ids, neg_extra) + \
                    (1.0 - bg_flag) * rpn_match

        target_gt_boxes = tf.gather(cur_gt_boxes, anchor_iou_argmax)
        # 计算anchor与对应gt_box偏移量
        rpn_bbox = box_refinement_graph(anchors, target_gt_boxes)
        rpn_bbox /= rpn_bbox_std_dev

        batch_rpn_match.append(rpn_match)
        batch_rpn_bbox.append(rpn_bbox)

    batch_rpn_match = tf.stack(batch_rpn_match, axis=0)
    batch_rpn_bbox = tf.stack(batch_rpn_bbox, axis=0)

    return batch_rpn_match, batch_rpn_bbox


class DetectionTargetLayer(tf.keras.layers.Layer):
    """ rpn数据打标层,把proposal输出的roi和gt box做偏差计算, 同时筛选出指定数量的样本和对应的目标, 作为损失计算用 """

    def __init__(self,
                 batch_size=2,
                 train_rois_per_image=200,
                 roi_poisitive_ratio=0.33,
                 bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                 use_mini_mask=True,
                 mask_shape=[28, 28]
                 ):
        super(DetectionTargetLayer, self).__init__()
        self.batch_size = batch_size
        self.train_rois_per_image = train_rois_per_image
        self.roi_poisitive_ratio = roi_poisitive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.mask_shape = mask_shape

    def call(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """

        :param proposals: [batch, N, 4]
        :param gt_class_ids: [batch, N]
        :param gt_boxes: [batch, N, 4]
        :param gt_masks: [batch, height, width, N]
        :return:
        """

        # Assertions
        # asserts = [
        #     tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
        #               name="roi_assertion"),
        # ]
        # with tf.control_dependencies(asserts):
        #     proposals = tf.identity(proposals)

        # Remove zero padding
        # proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
        # gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        # gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
        #                                name="trim_gt_class_ids")
        # gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
        #                      name="trim_gt_masks")

        final_rois = []
        final_roi_gt_class_ids = []
        final_deltas = []
        final_masks = []
        # print(proposals)
        # print(proposals[0])
        # print(proposals[1])
        for i in range(self.batch_size):
            cur_proposal = proposals[i]
            cur_gt_class_ids = gt_class_ids[i]
            cur_gt_boxes = gt_boxes[i]
            cur_gt_masks = gt_masks[i]

            # 去掉那些padding的数据
            cur_proposal, _ = trim_zeros_graph(cur_proposal, name="trim_proposals")
            cur_gt_boxes, non_zeros = trim_zeros_graph(cur_gt_boxes, name="trim_gt_boxes")
            cur_gt_class_ids = tf.boolean_mask(cur_gt_class_ids, non_zeros,
                                               name="trim_gt_class_ids")
            cur_gt_masks = tf.gather(cur_gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                                     name="trim_gt_masks")

            # Handle COCO crowds
            # A crowd box in COCO is a bounding box around several instances. Exclude
            # them from training. A crowd box is given a negative class ID.
            # crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
            non_crowd_ix = tf.where(cur_gt_class_ids > 0)[:, 0]
            # crowd_boxes = tf.gather(gt_boxes, crowd_ix)
            cur_gt_class_ids = tf.gather(cur_gt_class_ids, non_crowd_ix)
            cur_gt_boxes = tf.gather(cur_gt_boxes, non_crowd_ix)
            cur_gt_masks = tf.gather(cur_gt_masks, non_crowd_ix, axis=2)

            # Compute overlaps matrix [proposals, gt_boxes]
            overlaps = overlaps_graph(cur_proposal, cur_gt_boxes)

            # Compute overlaps with crowd boxes [proposals, crowd_boxes]
            # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
            # crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
            # no_crowd_bool = (crowd_iou_max < 0.001)

            # Determine positive and negative ROIs
            roi_iou_max = tf.reduce_max(overlaps, axis=1)
            # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            # positive_roi_bool = roi_iou_max >= 0.5
            positive_indices = tf.where(roi_iou_max >= 0.5)[:, 0]
            # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
            # negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]
            negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

            # Subsample ROIs. Aim for 33% positive
            # Positive ROIs
            # positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
            #                      config.ROI_POSITIVE_RATIO)
            positive_count = int(self.train_rois_per_image * self.roi_poisitive_ratio)
            positive_indices = tf.random.shuffle(positive_indices)[:positive_count]

            positive_count = tf.shape(positive_indices)[0]
            # Negative ROIs. Add enough to maintain positive:negative ratio.
            # r = 1.0 / config.ROI_POSITIVE_RATIO
            r = 1.0 / self.roi_poisitive_ratio
            negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
            negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

            # negative_count = self.train_rois_per_image - positive_count
            # negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
            # Gather selected ROIs
            positive_rois = tf.gather(cur_proposal, positive_indices)
            negative_rois = tf.gather(cur_proposal, negative_indices)
            # for i in negative_rois:
            #     print(i)

            # Assign positive ROIs to GT boxes.
            positive_overlaps = tf.gather(overlaps, positive_indices)
            roi_gt_box_assignment = tf.cond(
                tf.greater(tf.shape(positive_overlaps)[1], 0),
                true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
                false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
            )
            roi_gt_boxes = tf.gather(cur_gt_boxes, roi_gt_box_assignment)
            roi_gt_class_ids = tf.gather(cur_gt_class_ids, roi_gt_box_assignment)

            # Compute bbox refinement for positive ROIs
            deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
            # deltas /= config.BBOX_STD_DEV
            deltas /= self.bbox_std_dev

            # Assign positive ROIs to GT masks
            # Permute masks to [N, height, width, 1]
            transposed_masks = tf.expand_dims(tf.transpose(cur_gt_masks, [2, 0, 1]), -1)
            # Pick the right mask for each ROI
            roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
            #
            # # Compute mask targets
            boxes = positive_rois
            if self.use_mini_mask:
                # Transform ROI coordinates from normalized image space
                # to normalized mini-mask space.
                y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
                gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
                gt_h = gt_y2 - gt_y1
                gt_w = gt_x2 - gt_x1
                y1 = (y1 - gt_y1) / gt_h
                x1 = (x1 - gt_x1) / gt_w
                y2 = (y2 - gt_y1) / gt_h
                x2 = (x2 - gt_x1) / gt_w
                boxes = tf.concat([y1, x1, y2, x2], 1)
            box_ids = tf.range(0, tf.shape(roi_masks)[0])
            masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, self.mask_shape)
            # Remove the extra dimension from masks.
            masks = tf.squeeze(masks, axis=3)

            # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
            # binary cross entropy loss.
            masks = tf.round(masks)

            # Append negative ROIs and pad bbox deltas and masks that
            # are not used for negative ROIs with zeros.
            rois = tf.concat([positive_rois, negative_rois], axis=0)
            # for i in rois:
            #     print(i)
            N = tf.shape(negative_rois)[0]
            P = tf.maximum(self.train_rois_per_image - tf.shape(rois)[0], 0)
            rois = tf.pad(rois, [(0, P), (0, 0)])
            # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
            # 这里本来都是positive样本, 为了不同batch能stack到一起, 需要把负样本和不足200的样本padding进来(N+P)
            roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
            deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
            masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

            # 合并多个batch
            final_rois.append(rois)
            final_roi_gt_class_ids.append(roi_gt_class_ids)
            final_deltas.append(deltas)
            final_masks.append(masks)

        final_rois = tf.stack(final_rois, axis=0)
        final_roi_gt_class_ids = tf.stack(final_roi_gt_class_ids, axis=0)
        final_deltas = tf.stack(final_deltas, axis=0)
        final_masks = tf.stack(final_masks, axis=0)

        # return rois, roi_gt_class_ids, deltas, masks
        return final_rois, final_roi_gt_class_ids, final_deltas, final_masks


class PyramidROIAlignLayer(tf.keras.layers.Layer):
    """ 利用backbone网络输出的多层金字塔特征做相应的 roi特征截取 """

    def __init__(self, image_shape,batch_size, pool_shape=(7, 7)):
        super(PyramidROIAlignLayer, self).__init__()
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.pool_shape = pool_shape

    def log2_graph(self, x):
        """Implementation of Log2. TF doesn't have a native implementation."""
        return tf.math.log(x) / tf.math.log(2.0)

    def call(self, rois, mrcnn_feature_maps):

        # Assign each ROI to a level in the pyramid based on the ROI area.
        # y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        # image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)
        # TODO 修改那些计算log2_graph之前为0的roi,不放到后面计算了
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            # [[batch_idx, num_idx]]
            ix = tf.where(tf.equal(roi_level, level))
            # level_boxes = tf.gather_nd(boxes, ix)
            level_boxes = tf.gather_nd(rois, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                mrcnn_feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        # 这里根据batch,num_idx两个维度从小到大排序, 就是转成原来的排序
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        # 这里把rois的batch,num 拼接上pooled的 h,w,channels
        shape = tf.concat([tf.shape(rois)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)

        return pooled


class DetectionLayer(tf.keras.layers.Layer):
    """ 推理预测层 """

    def __init__(self, batch_size, bbox_std_dev, detection_max_instances, detection_nms_thres, detection_min_confidence):
        super(DetectionLayer, self).__init__()
        self.batch_size = batch_size
        self.bbox_std_dev = bbox_std_dev
        self.detection_max_instances = detection_max_instances
        self.detection_nms_thres = detection_nms_thres
        self.detection_min_confidence = detection_min_confidence


    def call(self, rois, probs, deltas, masks, window):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [batch, N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [batch, N, num_classes]. Class probabilities.
            deltas: [batch, N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            masks: [batch, N, h, w, num_classes]
            window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        final_detections = []
        final_masks = []
        for i in range(self.batch_size):

            cur_probs = probs[i]
            cur_rois = rois[i]
            cur_deltas = deltas[i]
            cur_mask = masks[i]

            non_zeros = tf.cast(tf.reduce_sum(tf.abs(cur_rois), axis=1), tf.bool)
            cur_rois = tf.boolean_mask(cur_rois, non_zeros)
            cur_deltas = tf.boolean_mask(cur_deltas, non_zeros)
            cur_probs = tf.boolean_mask(cur_probs, non_zeros)
            cur_mask = tf.boolean_mask(cur_mask, non_zeros)

            # Class IDs per ROI
            class_ids = tf.argmax(cur_probs, axis=1, output_type=tf.int32)
            # Class probability of the top class of each ROI
            # indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
            indices = tf.stack([tf.range(tf.shape(cur_probs)[0]), class_ids], axis=1)
            class_scores = tf.gather_nd(cur_probs, indices)
            # Class-specific bounding box deltas
            deltas_specific = tf.gather_nd(cur_deltas, indices)
            # Apply bounding box deltas
            # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
            refined_rois = apply_box_deltas_graph(cur_rois, deltas_specific * self.bbox_std_dev)
            # Clip boxes to image window
            refined_rois = clip_boxes_graph(refined_rois, window)

            # TODO: Filter out boxes with zero area
            # Filter out background boxes
            keep = tf.where(class_ids > 0)[:, 0]
            # Filter out low confidence boxes
            if self.detection_min_confidence:
                conf_keep = tf.where(class_scores >= self.detection_min_confidence)[:, 0]
                # keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
                # keep = tf.sparse_tensor_to_dense(keep)[0]
                keep = tf.sparse.to_dense(keep)[0]

            # Apply per-class NMS
            # 1. Prepare variables
            pre_nms_class_ids = tf.gather(class_ids, keep)
            pre_nms_scores = tf.gather(class_scores, keep)
            pre_nms_rois = tf.gather(refined_rois, keep)
            unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

            # 2. Map over class IDs
            def nms_keep_map(class_id):
                """Apply Non-Maximum Suppression on ROIs of the given class."""
                # Indices of ROIs of the given class
                ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
                # Apply NMS
                class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.detection_max_instances,
                    iou_threshold=self.detection_nms_thres)
                # Map indices
                class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
                # Pad with -1 so returned tensors have the same shape
                gap = self.detection_max_instances - tf.shape(class_keep)[0]
                class_keep = tf.pad(class_keep, [(0, gap)],
                                    mode='CONSTANT', constant_values=-1)
                # Set shape so map_fn() can infer result shape
                class_keep.set_shape([self.detection_max_instances])
                return class_keep

            # 2. Map over class IDs
            nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                                 dtype=tf.int64)
            # 3. Merge results into one list, and remove -1 padding
            nms_keep = tf.reshape(nms_keep, [-1])
            nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
            # 4. Compute intersection between keep and nms_keep
            # keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
            keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
            # keep = tf.sparse_tensor_to_dense(keep)[0]
            keep = tf.sparse.to_dense(keep)[0]
            # Keep top detections
            class_scores_keep = tf.gather(class_scores, keep)
            num_keep = tf.minimum(tf.shape(class_scores_keep)[0], self.detection_max_instances)
            top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
            keep = tf.gather(keep, top_ids)

            # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
            # Coordinates are normalized.
            detections = tf.concat([
                tf.gather(refined_rois, keep),
                # tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
                tf.cast(tf.gather(class_ids, keep),dtype=tf.float32)[..., tf.newaxis],
                tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)

            cur_mask = tf.gather(cur_mask, keep)
            # Pad with zeros if detections < DETECTION_MAX_INSTANCES
            gap = self.detection_max_instances - tf.shape(detections)[0]
            detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
            final_detections.append(detections)

            cur_mask = tf.pad(cur_mask,[(0,gap), (0,0), (0,0), (0,0)])
            final_masks.append(cur_mask)

        # [batch, N, (y1, x1, y2, x2, class_id, score)]
        final_detections = tf.stack(final_detections, axis=0)
        final_masks = tf.stack(final_masks, axis=0)
        return final_detections, final_masks