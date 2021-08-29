import sys

sys.path.append("../../mask_rcnn_in_tf2_keras")

import os
import math
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data.xml_ops import xml2dict
from PIL import Image


class VocSegmentDataGenerator:
    def __init__(self,
                 voc_data_path,
                 classes=['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                 is_training=True,
                 batch_size=2,
                 im_size=600,
                 max_instance=100,
                 data_max_size_per_class=150,
                 use_mini_mask=True,
                 mini_mask_shape=(56,56)):
        self.voc_data_path = voc_data_path
        self.classes = classes
        self.num_class = len(classes)
        self.segment_color_list = self._generate_color_list(self.num_class)
        self.class_color_map = self._generate_class_color_map()
        self.color_id_map = self._generate_color_id_map()
        self.is_training = is_training
        self.batch_size = batch_size
        self.im_size = im_size
        self.max_instance = max_instance
        self.data_max_size_per_class = data_max_size_per_class
        self.use_mini_mask = use_mini_mask
        self.mini_mask_shape = mini_mask_shape

        # 加载文件
        self.total_batch_size = 0
        self.current_batch_index = 0
        self.img_files = []
        self.cls_mask_files = []
        self.obj_mask_files = []
        self.anno_files = []
        self.file_indices = []
        self.__load_files()
        self.__balance_class_data()

    def _generate_color_list(self, N, normalized=False):
        """生成voc颜色数组"""

        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap

    def _generate_class_color_map(self):
        """生成类别对应的RGB数组"""
        color_list = self._generate_color_list(self.num_class)
        return dict(zip(self.classes, color_list))

    def _generate_color_id_map(self):
        """生成颜色和类别索引的映射字典, key为颜色数组hash结果"""
        color_list = self._generate_color_list(self.num_class)
        color_id_map = {}
        for i, color in enumerate(color_list):
            color_id_map[hash(np.array(color, dtype=np.int8).data.tobytes())] = i
        return color_id_map

    def _get_color_idx(self, color):
        """根据颜色数组拿到对应类别, 颜色数组计算相应hash key, 没有返回-1"""
        hask_key = hash(np.array(color, dtype=np.int8).data.tobytes())
        return self.color_id_map.get(hask_key, -1)

    def _get_mask(self, cls_file, obj_file):
        """ 根据类别RGB数组截取对应box内目标分割mask形状
        :param class_name:
        :param bdbox:
        :param im:
        :param segement_im:
        :return: [{class_idx, mask, bdbox}]
        """
        # seg_obj_f = os.path.join(self.voc_data_path, "SegmentationObject", file_name + '.png')
        # seg_cls_f = os.path.join(self.voc_data_path, "SegmentationClass", file_name + '.png')

        # cls图片直接转rgb数组
        seg_cls_im = cv2.imread(cls_file)
        seg_cls_im = cv2.cvtColor(seg_cls_im, cv2.COLOR_BGR2RGB)

        # Image.open 使用p模式打开, 再拿到各个目标的RBG
        seg_obj_im = Image.open(obj_file)
        # 这里拿到调试色板, 对应图片出现的各个RGB颜色数组
        obj_im_color_palette = np.array(seg_obj_im.getpalette()).reshape(256, 3)
        # 这里拿到图片里面出现的各个目标对应调色板上的颜色索引
        obj_ids = np.unique(np.array(seg_obj_im))
        # 这里拿到各个目标的颜色
        obj_id_colors = [obj_im_color_palette[i] for i in obj_ids if i != 0 and i != 255]
        # 分别拿到各个目标的mask
        obj_im_rgb = np.array(seg_obj_im.convert('RGB'))
        objs_mask_cls_bdbox = []
        for i, obj_color in enumerate(obj_id_colors):
            obj_mask = np.logical_and(np.logical_and(obj_im_rgb[:, :, 0] == obj_color[0],
                                                     obj_im_rgb[:, :, 1] == obj_color[1]),
                                      obj_im_rgb[:, :, 2] == obj_color[2])
            obj_mask = np.array(obj_mask, dtype=np.int8)
            # 这里截取类别图里对应的每个mask
            cls_mask = np.array(seg_cls_im * np.tile(np.expand_dims(obj_mask, -1), 3), dtype=np.uint8)
            # 找到每个mask对应的color索引
            cls_mask_reshpe = np.reshape(cls_mask, (-1, 3))
            color_unique_rgb_list = np.unique(cls_mask_reshpe, axis=0)
            for c in color_unique_rgb_list:
                color_idx = self._get_color_idx(c)
                # 0为背景, -1为无效颜色数据
                if color_idx != 0 and color_idx != -1:
                    cls_mask_bool = np.logical_and(np.logical_and(cls_mask[:, :, 0] == c[0],
                                                                  cls_mask[:, :, 1] == c[1]),
                                                   cls_mask[:, :, 2] == c[2])
                    cls_mask_bool = np.array(cls_mask_bool, dtype=np.int8)
                    h, w = np.shape(cls_mask_bool)
                    # 计算bdbox
                    rows, cols = np.where(cls_mask_bool)
                    # [ymin, xmin, ymax, xmax]
                    xmin = np.min(cols)# - 2 if np.min(cols) - 2 >= 0 else np.min(cols)
                    ymin = np.min(rows)# - 2 if np.min(rows) - 2 >= 0 else np.min(rows)
                    xmax = np.max(cols)# + 2 if np.max(cols) + 2 <= w else np.max(cols)
                    ymax = np.max(rows)# + 2 if np.max(rows) + 2 <= h else np.max(rows)
                    bdbox = [ymin, xmin, ymax, xmax]
                    # 判断bdbox是否有效
                    if bdbox[2] > bdbox[0] and bdbox[3] > bdbox[1]:
                        objs_mask_cls_bdbox.append({
                            "class_idx": color_idx,
                            "mask": cls_mask_bool,
                            "bdbox": bdbox
                        })
        return objs_mask_cls_bdbox

    def __load_files(self):
        """加载训练文件"""

        if self.is_training:
            file = os.path.join(self.voc_data_path, "ImageSets/Segmentation", "train.txt")
        else:
            file = os.path.join(self.voc_data_path, "ImageSets/Segmentation", 'val.txt')

        with open(file, encoding='utf-8', mode='r') as f:
            data = f.readlines()
            for file_name in data:
                file_name = file_name.strip()
                img_file_jpg = os.path.join(self.voc_data_path, "JPEGImages", file_name + '.jpg')
                cls_mask_png = os.path.join(self.voc_data_path, "SegmentationClass", file_name + '.png')
                obj_mask_png = os.path.join(self.voc_data_path, "SegmentationObject", file_name + '.png')
                anno_xml = os.path.join(self.voc_data_path, "Annotations", file_name + '.xml')

                if os.path.isfile(img_file_jpg) and os.path.isfile(cls_mask_png) and os.path.isfile(obj_mask_png):
                    self.img_files.append(img_file_jpg)
                    self.cls_mask_files.append(cls_mask_png)
                    self.obj_mask_files.append(obj_mask_png)
                    self.anno_files.append(anno_xml)

        self.total_batch_size = int(math.floor(len(self.img_files) / self.batch_size))
        self.file_indices = np.arange(len(self.img_files))

    def __balance_class_data(self):
        """ 平衡每个类别样本数 """

        # self.file_indices = np.arange(len(self.anno_files))
        # np.random.shuffle(self.file_indices)

        # balance_annotation_files = []
        # balance_img_files = []
        balance_file_indices = []
        per_class_nums = dict(zip(self.classes, [0] * len(self.classes)))

        for i in self.file_indices:
            annotation = xml2dict(self.anno_files[i])
            objs = annotation['annotation']['object']

            all_classes = []
            if type(objs) == list:
                for obj in objs:
                    all_classes.append(obj['name'])
            else:
                all_classes.append(objs['name'])

            keep = False
            # if 'person' not in all_classes:
            for cls in set(all_classes):
                if per_class_nums[cls] <= self.data_max_size_per_class:
                    keep = True
                    per_class_nums[cls] += 1
            if keep:
                balance_file_indices.append(i)
                # balance_annotation_files.append(self.annotation_files[i])
                # balance_img_files.append(self.img_files[i])

        remove_file_nums = len(self.anno_files) - len(balance_file_indices)
        # remove_file_nums = len(self.file_indices) - len(balance_file_indices)
        # self.annotation_files = balance_annotation_files
        # self.img_files = balance_img_files

        # self.total_batch_size = int(math.floor(len(self.annotation_files) / self.batch_size))
        self.total_batch_size = int(math.floor(len(balance_file_indices) / self.batch_size))
        # self.file_indices = np.arange(len(self.annotation_files))
        self.file_indices = balance_file_indices
        print("after balance total file nums: {}, remove {} files".format(len(balance_file_indices), remove_file_nums))
        print("every class nums: {}".format(per_class_nums))

    def check_mask_obj_class(self):
        """ 检查训练的mask目标样本分布情况 """
        clss = []
        areas = []
        objs_per_img = []
        for anno_xml in self.anno_files:
            anno = xml2dict(anno_xml)
            objs = anno['annotation']['object']
            num_obj = 0
            if type(objs) == list:
                for box in objs:
                    cls = box['name']
                    xmin = int(float(box['bndbox']['xmin']))
                    ymin = int(float(box['bndbox']['ymin']))
                    xmax = int(float(box['bndbox']['xmax']))
                    ymax = int(float(box['bndbox']['ymax']))
                    area = (ymax - ymin) * (xmax - xmin)
                    clss.append(cls)
                    areas.append(area)
                    num_obj += 1
            else:
                cls = objs['name']
                xmin = int(float(objs['bndbox']['xmin']))
                ymin = int(float(objs['bndbox']['ymin']))
                xmax = int(float(objs['bndbox']['xmax']))
                ymax = int(float(objs['bndbox']['ymax']))
                area = (ymax - ymin) * (xmax - xmin)
                clss.append(cls)
                areas.append(area)
                num_obj += 1
            objs_per_img.append(num_obj)



        cls_counter = Counter(clss)
        keys = list(Counter(cls_counter).keys())
        values = list(Counter(cls_counter).values())
        plt.bar(x=keys, height=values, width=0.8, alpha=0.8, color='red', label='class count')
        plt.xticks(keys, keys, rotation=90)
        for a, b in zip(keys, values):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
        plt.savefig('./tmp/voc_class_count.png', bbox_inches='tight')
        # print(Counter(areas))

        # gt_box面积分布
        plt.cla()
        areas = np.array(areas, dtype=np.int32)
        plt.hist(areas, bins=60, facecolor="red", edgecolor="black", alpha=0.7)
        plt.xlabel("groud true bounding box areas")
        # 显示纵轴标签
        plt.ylabel("freq")
        # 显示图标题
        plt.title("gt bbox area hist")
        plt.savefig('./tmp/gt_box_hist.png')

        # area < 50000
        plt.cla()
        areas = np.array(areas, dtype=np.int32)
        areas = areas[areas < 50000]
        plt.hist(areas, bins=60, facecolor="red", edgecolor="black", alpha=0.7)
        plt.xlabel("groud true bounding box areas")
        # 显示纵轴标签
        plt.ylabel("freq")
        # 显示图标题
        plt.title("gt bbox area(< 50000) hist")
        plt.savefig('./tmp/gt_box_hist_less_50000.png')

        # area < 10000
        plt.cla()
        areas = np.array(areas, dtype=np.int32)
        areas = areas[areas < 10000]
        plt.hist(areas, bins=60, facecolor="red", edgecolor="black", alpha=0.7)
        plt.xlabel("groud true bounding box areas")
        # 显示纵轴标签
        plt.ylabel("freq")
        # 显示图标题
        plt.title("gt bbox area(< 10000) hist")
        plt.savefig('./tmp/gt_box_hist_less_10000.png')

        # 每张图片的目标数
        plt.cla()
        objs_per_img = np.array(objs_per_img, dtype=np.int32)
        # print(Counter(objs_per_img))
        objs_per_img = objs_per_img[objs_per_img < 10]
        plt.hist(objs_per_img, bins=60, facecolor="red", edgecolor="black", alpha=0.7)
        plt.xlabel("object nums per image")
        # 显示纵轴标签
        plt.ylabel("freq")
        # 显示图标题
        plt.title("objs hist")
        plt.savefig('./tmp/objs_per_img.png')

    def _on_epoch_end(self):
        """混洗数据索引"""
        self.file_indices = np.arange(len(self.img_files))
        np.random.shuffle(self.file_indices)

    def next_batch(self):
        """每次迭代生成的batch文件"""
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()
            self.__balance_class_data()

        indices = self.file_indices[self.current_batch_index * self.batch_size:
                                    (self.current_batch_index + 1) * self.batch_size]
        # cur_img_files = ["../../data/VOCdevkit/VOC2012/JPEGImages/2007_000243.jpg","../../data/VOCdevkit/VOC2012/JPEGImages/2007_000243.jpg"]
        # cur_cls_files = ["../../data/VOCdevkit/VOC2012/SegmentationClass/2007_000243.png","../../data/VOCdevkit/VOC2012/SegmentationClass/2007_000243.png"]
        # cur_obj_files = ["../../data/VOCdevkit/VOC2012/SegmentationObject/2007_000243.png","../../data/VOCdevkit/VOC2012/SegmentationObject/2007_000243.png"]
        cur_img_files = [self.img_files[k] for k in indices]
        cur_cls_files = [self.cls_mask_files[k] for k in indices]
        cur_obj_files = [self.obj_mask_files[k] for k in indices]
        print(cur_img_files)
        print(cur_cls_files)
        print(cur_obj_files)
        imgs, masks, gt_boxes, labels = self._data_generation(img_files=cur_img_files,
                                                              cls_files=cur_cls_files,
                                                              obj_files=cur_obj_files)
        self.current_batch_index += 1
        return imgs, masks, gt_boxes, labels

    def _data_generation(self, img_files, cls_files, obj_files):
        """ 数据生成
        :param img_files:
        :param cls_files:
        :param obj_files:
        :return:
        """
        imgs = []
        labels = []
        gt_boxes = []
        masks = []
        for i in range(len(img_files)):
            img = cv2.imread(img_files[i])
            # [{'class_idx':, 'mask':, 'bdbox':}]
            objs_mask_cls_bdbox = self._get_mask(cls_files[i], obj_files[i])
            label = [obj['class_idx'] for obj in objs_mask_cls_bdbox]
            gt_box = [obj['bdbox'] for obj in objs_mask_cls_bdbox]
            mask = [obj['mask'] for obj in objs_mask_cls_bdbox]

            if img is not None and len(label) > 0 and len(gt_box) > 0 and mask is not None:
                im_resize, masks_resize, gt_box = self._resize_im(img, mask)
                if self.use_mini_mask:
                    masks_resize = self._resise_mini_mask(masks_resize, gt_box)

                # 输出一些方便调试的日志信息
                print(img_files[i])
                print(cls_files[i])
                print(obj_files[i])
                print("img shape: ", np.shape(im_resize))
                print("mask shape: ", np.shape(masks_resize))
                print("label: ", label)
                print("gt_box: \n", gt_box)

                # 填补保证支持多个batch
                masks_instances = np.shape(masks_resize)[-1]
                if masks_instances >= self.max_instance:
                    masks_resize = masks_resize[:, :, :self.max_instance]
                    label = label[:self.max_instance]
                    gt_box = gt_box[:self.max_instance, :]
                else:
                    padding_size = self.max_instance - masks_instances
                    masks_resize = np.pad(masks_resize, [[0, 0], [0, 0], [0, padding_size]])
                    label = np.pad(label, [0, padding_size])
                    gt_box = np.pad(gt_box, [[0, padding_size], [0, 0]])

                imgs.append(im_resize)
                masks.append(masks_resize)
                gt_boxes.append(gt_box)
                labels.append(label)

        imgs = np.array(imgs, dtype=np.int8)
        masks = np.array(masks, dtype=np.int8)
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)

        return imgs - np.array([[[102.9801, 115.9465, 122.7717]]]), masks, gt_boxes, labels

    def _resize_im(self, origin_im, mask_im):
        """ 对图片/mask/box resize

        :return im_blob: [h, w, 3]
                masks_resize: [h, w, instance_num]
                gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        im_shape = np.shape(origin_im)
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.im_size) / float(im_size_max)

        # resize原始图片
        im_resize = cv2.resize(origin_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_resize_shape = np.shape(im_resize)
        im_blob = np.zeros((self.im_size, self.im_size, 3), dtype=np.float32)
        im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize

        # resize mask/box
        gt_boxes = []
        masks_resize = []
        for m in mask_im:
            m = np.array(m, dtype=np.float32)
            m_resize = cv2.resize(m, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            m_resize = np.array(m_resize >= 0.5, dtype=np.int8)

            # 计算bdbox
            h, w = np.shape(m_resize)
            rows, cols = np.where(m_resize)
            # [xmin, ymin, xmax, ymax]
            xmin = np.min(cols)# - 2 if np.min(cols) - 2 >= 0 else np.min(cols)
            ymin = np.min(rows)# - 2 if np.min(rows) - 2 >= 0 else np.min(rows)
            xmax = np.max(cols)# + 2 if np.max(cols) + 2 <= w else np.max(cols)
            ymax = np.max(rows)# + 2 if np.max(rows) + 2 <= h else np.max(rows)
            bdbox = [ymin, xmin, ymax, xmax]
            gt_boxes.append(bdbox)

            mask_blob = np.zeros((self.im_size, self.im_size, 1), dtype=np.float32)
            mask_blob[0:im_resize_shape[0], 0:im_resize_shape[1], 0] = m_resize
            masks_resize.append(mask_blob)

        # [instance_num, [ymin, xmin, ymax, xmax]]
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        # [h, w, instance_num]
        masks_resize = np.concatenate(masks_resize, axis=-1)

        return im_blob, masks_resize, gt_boxes

    def _resise_mini_mask(self,masks, boxes):
        """  mask处理成最小mask """
        mini_masks = []
        h,w,c = np.shape(masks)
        for i in range(c):
            ymin, xmin, ymax, xmax = boxes[i]
            mask = masks[int(ymin):int(ymax), int(xmin):int(xmax), i]
            mini_m = cv2.resize(mask, self.mini_mask_shape, interpolation=cv2.INTER_LINEAR)
            mini_m = np.array(mini_m >= 0.5, dtype=np.int8)
            mini_m = np.expand_dims(mini_m,axis=-1)
            mini_masks.append(mini_m)
        mini_masks = np.concatenate(mini_masks, axis=-1)
        return mini_masks


if __name__ == "__main__":
    import cv2
    from PIL import Image
    from data.visual_ops import draw_instance

    vsg = VocSegmentDataGenerator("./VOCdevkit/VOC2012", batch_size=2)
    # vsg.check_mask_obj_class()
    imgs, masks, gt_boxes, labels = vsg.next_batch()
    # print(np.shape(masks),np.shape(gt_boxes),np.shape(labels))
    # cv2.imshow('', np.array(masks[2,:,:,0] * 255,dtype=np.uint8))
    # cv2.waitKey(0)
    # active_num = len(np.where(labels)[1])
    # # img = np.array(imgs[0],dtype=np.uint8)
    # mask = masks[1]
    # img_mask = draw_instance(imgs[1], mask[:,:,:active_num])
    # img_mask = np.array(img_mask, dtype=np.uint8)
    # # im = Image.fromarray(img_mask[:,:,::-1])
    # # im.show()
    # cv2.imshow("",img_mask)
    # cv2.waitKey(0)

