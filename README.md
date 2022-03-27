## Mask RCNN in tf2-keras

### bilibili讲解视频

[![Watch the video](https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/b站视频页.png)](https://www.bilibili.com/video/BV1qA411w7Zg?share_source=copy_web)

### requirements

- tensorflow-gpu >= 2.1.0
- xmltodict
- Pillow
- opencv-python
- matplotlib

### 检测效果

- VOC2012

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample1.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample2.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample4.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample5.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample6.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample7.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/sample8.png" width="350" height="230"/>


### 训练`Voc2012`或者`CoCo`数据

0. 训练voc数据
```python
python3 train_voc.py
```

1. 训练coco数据
```python
python3 train_coco.py
```

2. tensorboard查看效果
```python
tensorboard --host 0.0.0.0 --logdir ./logs/ --port 9013 --samples_per_plugin=images=40
```

3. 浏览器打开: `http://127.0.0.1:9013`

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/scalar.png" width="800" height="437"/>  

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/mask_rcnn/images.png" width="800" height="437"/>


### 测试`Voc2012`
1. 构建模型
```python
from mrcnn.mask_rcnn import MaskRCNN
mrcnn = MaskRCNN(classes=['_background_', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
                 is_training=False,
                 batch_size=1,
                 image_shape=[640,640,3])
```

2. 加载权重
```python
model_path = '.h5 file path'
mrcnn.load_weights(model_path, by_name=True)
```

3. 测试, 在`/tmp`目录下可以看到检测结果保存的图片
```python
import cv2
import numpy as np
from mrcnn.anchors_ops import get_anchors
anchors = get_anchors(image_shape=mrcnn.image_shape,
                      scales=mrcnn.scales,
                      ratios=mrcnn.ratios,
                      feature_strides=mrcnn.feature_strides,
                      anchor_stride=mrcnn.anchor_stride)
all_anchors = np.stack([anchors], axis=0)

image = cv2.imread("image_path")
image = cv2.resize(image, (mrcnn.image_shape[0], mrcnn.image_shape[1]))
image = np.stack([image], axis=0)

boxes,class_ids,scores,masks = mrcnn.predict(image=image, anchors=all_anchors, draw_detect_res_figure=True)
```

### 训练自己的数据

1. labelme打标好自己的数据
2. 打开`data/labelme2coco.py`脚本, 修改如下地方
```angular2html
input_dir = '这里写labelme打标时保存json标记文件的目录'
output_dir = '这里写要转CoCo格式的目录，建议建一个空目录'
labels = "这里是你打标时所有的类别名, txt文本即可, 注意第一个类名是'_background_', 剩下的都是你打标的类名"
```
3. 执行`data/labelme2coco.py`脚本会在`output_dir`生成对应的json文件和图片
4. 修改`train_coco.py`文件中`classes`和`coco_annotation_file`, 注意`classes`第一个需要是'\_background\_', 每个类名单独一行
5. 开始训练, `python3 train_coco.py`

### 代码细节

- 源码训练voc2012数据: /official_mask_rcnn/samples/voc/voc2012.py
- 复现代码: /mrcnn
- 数据生成: /data/generate_voc_segment_data.py