import hashlib
import json
import os
import pandas as pd
import shutil
import yaml

from sklearn.model_selection import train_test_split
from .yolo_converter import YOLOAnnotationConverter

class DataFunctions():
    def __init__(self, local_yolo_dir, to_name='image', from_name='label', label_type='bbox'):
        self.yolo_conv = YOLOAnnotationConverter(
            dataset_dir=local_yolo_dir,
            classes=[],
            to_name=to_name, 
            from_name=from_name,
            label_type=label_type)

    def remove_yolo_v8_labels(self):
        labels = os.path.join(self.yolo_conv.dataset_dir, 'labels')
        shutil.rmtree(labels, ignore_errors=True)
    
    def remove_yolo_v8_dataset(self):
        shutil.rmtree(self.yolo_conv.dataset_dir, ignore_errors=True)
        if os.path.exists('custom_yolo.yaml'):
            os.remove('custom_yolo.yaml')

    def create_yolo_v8_dataset_yaml(self, dataset, download=True, label_field  ="annotation"):
        path = os.path.abspath(self.yolo_conv.dataset_dir)

        if download:
            self.remove_yolo_v8_dataset()
            for split in ('train', 'valid', 'test'):
                split_ds = dataset[dataset['split'] == split]
                target_dir = os.path.join(path, f'images/{split}')
                # 先下载（会创建嵌套）
                split_ds.all().download_files(target_dir=target_dir, keep_source_prefix=False)
                # 处理可能的嵌套结构
                for root, dirs, files in os.walk(target_dir):
                    if root != target_dir:  # 如果不是目标目录本身
                        for file in files:
                            src = os.path.join(root, file)
                            dst = os.path.join(target_dir, file)
                            if not os.path.exists(dst):  # 避免覆盖
                                shutil.move(src, dst)
                # 清理空目录
                for root, dirs, files in os.walk(target_dir, topdown=False):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if not os.listdir(dir_path):  # 如果目录为空
                            os.rmdir(dir_path)
        else:
            self.remove_yolo_v8_labels()

        # 标签转换
        for dp in dataset.all().get_blob_fields(label_field):
            self.yolo_conv.from_de(dp, label_field)

        train = 'images/train'
        val = 'images/valid'
        test = 'images/test'

        yaml_dict = {
            'path': path, 
            'train': train, 
            'val': val,
            'test': test,
            'names': {i: name for i, name in enumerate(self.yolo_conv.classes)}
        }
        with open("custom_yolo.yaml", "w") as file:
            file.write(yaml.dump(yaml_dict))

