import os
import sys
import math
import random
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from official_mask_rcnn.mrcnn.config import Config
from official_mask_rcnn.mrcnn import utils
from xml_ops import xml2dict


class Voc2012Config(Config):
    # Give the configuration a recognizable name
    NAME = "voc2012"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 700

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class Voc2012Dataset(utils.Dataset):
    """ 继承Dataset类, 重写load_image, image_reference, load_mask三个方法
    """
    def __init__(self):
        super(Voc2012Dataset, self).__init__()
        self.color_id_map = {}

    def load_voc2012(self, voc_data_root_path, is_training):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        for i, c in enumerate(classes):
            self.add_class("voc2012", i + 1, c)

        # Add images
        if is_training:
            file = os.path.join(voc_data_root_path, "ImageSets/Segmentation", "train.txt")
        else:
            file = os.path.join(voc_data_root_path, "ImageSets/Segmentation", 'val.txt')

        with open(file, encoding='utf-8', mode='r') as f:
            data = f.readlines()
            i = 0
            for file_name in data:
                file_name = file_name.strip()
                img_file_jpg = os.path.join(voc_data_root_path, "JPEGImages", file_name + '.jpg')
                cls_mask_png = os.path.join(voc_data_root_path, "SegmentationClass", file_name + '.png')
                obj_mask_png = os.path.join(voc_data_root_path, "SegmentationObject", file_name + '.png')
                anno_xml = os.path.join(voc_data_root_path, "Annotations", file_name + '.xml')

                if os.path.isfile(img_file_jpg) and os.path.isfile(cls_mask_png) and os.path.isfile(obj_mask_png):
                    self.add_image(source="voc2012", image_id=i, path=None, img_file_path=img_file_jpg,
                                   cls_mask_path=cls_mask_png, obj_mask_path=obj_mask_png, anno_xml_path=anno_xml)
                    i += 1

        self.color_id_map = self._generate_color_id_map()

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img_file_path = info['img_file_path']
        origin_im = cv2.imread(img_file_path)

        # im_shape = np.shape(origin_im)
        # im_size_max = np.max(im_shape[0:2])
        # im_scale = float(Voc2012Config.IMAGE_MAX_DIM) / float(im_size_max)
        #
        # # resize原始图片
        # im_resize = cv2.resize(origin_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_resize_shape = np.shape(im_resize)
        # im_blob = np.zeros((Voc2012Config.IMAGE_MAX_DIM, Voc2012Config.IMAGE_MAX_DIM, 3), dtype=np.float32)
        # im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize
        # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        # image = image * bg_color.astype(np.uint8)
        # for shape, color, dims in info['shapes']:
        #     image = self.draw_shape(image, shape, dims, color)
        # return im_blob
        return origin_im

    def image_reference(self, image_id):
        """Return the voc2012 data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "voc2012":
            return info["img_file_path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        cls_file = info['cls_mask_path']
        obj_file = info['obj_mask_path']
        mask_cls_box = self._get_mask(cls_file, obj_file)
        masks = []
        classes = []
        boxes = []
        for mcb in mask_cls_box:
            masks.append(np.expand_dims(mcb['mask'],axis=-1))
            classes.append(mcb['class_idx'])
            boxes.append(mcb['bdbox'])

        classes = np.array(classes, dtype=np.int8)
        masks = np.concatenate(masks, axis=-1)
        boxes = np.array(boxes,dtype=np.int32)

        # shapes = info['shapes']
        # count = len(shapes)
        # mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        # for i, (shape, _, dims) in enumerate(info['shapes']):
        #     mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
        #                                           shape, dims, 1)
        # # Handle occlusions
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(count - 2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(
        #         occlusion, np.logical_not(mask[:, :, i]))
        # # Map class names to class IDs.
        # class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return masks, classes

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
        color_list = self._generate_color_list(len(self.class_info))
        classes = list(map(lambda x:x['name'], self.class_info))
        return dict(zip(classes, color_list))

    def _generate_color_id_map(self):
        """生成颜色和类别索引的映射字典, key为颜色数组hash结果"""
        color_list = self._generate_color_list(len(self.class_info))
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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

if __name__ == '__main__':
    from official_mask_rcnn.mrcnn import visualize
    import official_mask_rcnn.mrcnn.model as modellib
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    vc = Voc2012Config()
    vc.display()

    dataset_train = Voc2012Dataset()
    dataset_train.load_voc2012(voc_data_root_path='../../../data/voc2012_46_samples', is_training=True)
    dataset_train.prepare()

    dataset_val = Voc2012Dataset()
    dataset_val.load_voc2012(voc_data_root_path='../../../data/voc2012_46_samples', is_training=False)
    dataset_val.prepare()

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode="training", config=vc,
                              model_dir='../../logs')
    model.load_weights("../../mask_rcnn_coco.h5", by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val,
                learning_rate=0.0001,
                epochs=100,
                layers="all")


    class InferenceConfig(Voc2012Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir='../../logs')

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())
