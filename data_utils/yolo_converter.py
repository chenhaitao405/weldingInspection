import os
import json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

from PIL import Image

from label_studio_converter.imports.label_config import generate_label_config

from .dags_converter import DagsConverter

logger = logging.getLogger('root')

class YOLOAnnotationConverter():
    def __init__(self, dataset_dir, classes = [], to_name='image', from_name='label', label_type='bbox'):
        """Instantiate YOLO Annotation Convertert
        """
        self.ann_type = "YOLO"
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.url_col = "dagshub_download_url"
        self.to_name = to_name
        self.from_name = from_name
        self.label_type = label_type
        
        # build categories=>labels dict
        if len(self.classes) == 0:
            if not self._update_classes_from_file():
                logger.warning(
                    'No classes.txt file found and now classes array supplied,'
                    'this might result in errors due to missing classes'
                )
        
        categories = {i: line for i, line in enumerate(self.classes)}
        logger.info(f'Found {len(categories)} categories')

        # generate and save labeling config
        self.config = generate_label_config(
            categories,
            {from_name: 'RectangleLabels'},
            to_name,
            from_name
        )
        
        self.ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)
    
    def _update_classes_from_file(self):
        notes_file = os.path.join(self.dataset_dir, 'classes.txt')
        if os.path.exists(notes_file):
            with open(notes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
            return True
        return False

    def _create_bbox(self, line, image_width, image_height):
        label_id, x, y, width, height = line.split()
        x, y, width, height = (
            float(x),
            float(y),
            float(width),
            float(height),
        )
        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "rectanglelabels",
            "value": {
                "x": (x - width / 2) * 100,
                "y": (y - height / 2) * 100,
                "width": width * 100,
                "height": height * 100,
                "rotation": 0,
                "rectanglelabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item
    
    def _create_segmentation(self, line, image_width, image_height):
        label_id = line.split()[0]
        points = [[float(x[0]), float(x[1])] for x in zip(*[iter(line.split()[1:])]*2)]

        for i in range(len(points)):
            points[i][0] = points[i][0] * 100.0
            points[i][1] = points[i][1] * 100.0

        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "polygonlabels",
            "value": {
                "closed": True,
                "points": points,
                "polygonlabels": [self.classes[int(label_id)]],
            },
            "to_name": self.to_name,
            "from_name": self.from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height,
        }
        return item
                        
    def to_de(self, row, out_type="annotations"):
        """Convert YOLO labeling to Label Studio JSON
        :param out_type: annotation type - "annotations" or "predictions"
        """
        # define coresponding label file and check existence
        image_path = row["path"]
        if not "images/" in image_path:
            image_path = os.path.join("images", image_path)
        label_path = image_path.replace(image_path.split(".")[-1], "txt")
        if "/images/" in label_path or label_path.startswith("images/"):
            label_path = label_path.replace("images/","labels/")
        else:
            label_path = os.path.join("labels", label_path)
    
        label_file = os.path.join(self.dataset_dir, label_path)
        image_file = os.path.join(self.dataset_dir, image_path)
        image_width = 0
        image_height = 0
    
        task = None
                                  
        if os.path.exists(label_file):
            task = {
                "data": {
                    # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                    "image": row[self.url_col]
                }
            }
                                  
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if not (image_width and image_height):
                # default to opening file if we aren't given image dims. slow!
                with Image.open(os.path.join(image_file)) as im:
                    image_width, image_height = im.size

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    if 'bbox' in self.label_type:
                        item = self._create_bbox(line, image_width, image_height)
                        task[out_type][0]['result'].append(item)
                    if 'segmentation' in self.label_type:
                        item = self._create_segmentation(line, image_width, image_height)
                        task[out_type][0]['result'].append(item)
                        task['is_labeled'] = True

        if task:
            return json.dumps(task).encode()

    def from_de(self, row, label_field="annotation"):
        import os

        # 获取标注数据
        annotations_data = row[label_field]

        # 检查是否是 MetadataAnnotations 对象
        if hasattr(annotations_data, '__class__') and 'MetadataAnnotations' in str(type(annotations_data)):
            # 直接转换为 YOLO 格式

            # 确定标签文件路径
            image_path = row.path  # 直接访问属性
            split = row.metadata.get('split')  # 从metadata字典获取

            # 创建标签目录
            label_dir = os.path.join(self.dataset_dir, "labels", split)
            os.makedirs(label_dir, exist_ok=True)

            # 生成标签文件名
            filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, filename + '.txt')

            # 准备YOLO格式的标注
            yolo_annotations = []

            for ann in annotations_data.annotations:
                # 获取类别名称和ID
                for category_name in ann.categories.keys():
                    # 确保类别在类别列表中
                    if category_name not in self.classes:
                        self.classes.append(category_name)

                    category_id = self.classes.index(category_name)

                    # 构建YOLO格式的多边形标注
                    # ann.points 已经是归一化坐标 (0-1)
                    points = []
                    for point in ann.points:
                        points.extend([point.x, point.y])

                    # YOLO格式：类别ID + 所有点的坐标
                    yolo_annotation = [category_id] + points
                    yolo_annotations.append(yolo_annotation)

            # 写入标签文件
            with open(label_path, 'w') as f:
                for annotation in yolo_annotations:
                    # 将所有值转换为字符串并用空格连接
                    line = ' '.join(str(val) for val in annotation)
                    f.write(line + '\n')

        else:
            # 原有的处理逻辑（针对其他格式）
            ls_converter = DagsConverter(self.config, self.dataset_dir, download_resources=False)
            ls_converter.convert_to_yolo(input_data=annotations_data,
                                         output_dir=self.dataset_dir,
                                         output_label_dir=os.path.join(self.dataset_dir, "labels",
                                                                       row.metadata['split'],
                                                                       *(row.path.split("/")[:-1])),
                                         is_dir=False)

            # 更新类别
            ls_converter._get_labels()
            self._update_classes_from_file()
