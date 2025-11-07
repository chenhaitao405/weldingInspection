import hashlib
import json
import os
import pandas as pd
import shutil
import yaml
import random

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

    def add_split_field(self, datasource, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        给数据集添加 split 字段，将数据分为 train, valid, test 三个部分

        Parameters:
        -----------
        datasource : DagsHub datasource object
            数据源对象
        train_ratio : float
            训练集比例，默认 0.7
        valid_ratio : float
            验证集比例，默认 0.15
        test_ratio : float
            测试集比例，默认 0.15
        random_seed : int
            随机种子，用于确保可重复性

        Returns:
        --------
        dict : 包含各个split数量的统计信息
        """
        # 验证比例和为1
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 0.001, \
            "Split ratios must sum to 1.0"

        # 设置随机种子
        random.seed(random_seed)

        # 获取所有数据点
        print("Fetching all datapoints...")
        all_datapoints = datasource.all()
        datapoint_list = list(all_datapoints)
        total_count = len(datapoint_list)

        print(f"Total datapoints: {total_count}")

        # 生成随机索引并打乱
        indices = list(range(total_count))
        random.shuffle(indices)

        # 计算各个集合的大小
        train_size = int(total_count * train_ratio)
        valid_size = int(total_count * valid_ratio)
        test_size = total_count - train_size - valid_size  # 确保所有数据都被分配

        # 分割索引
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        # 创建索引到split的映射
        split_mapping = {}
        for idx in train_indices:
            split_mapping[idx] = 'train'
        for idx in valid_indices:
            split_mapping[idx] = 'valid'
        for idx in test_indices:
            split_mapping[idx] = 'test'

        # 使用 metadata_context 批量更新metadata
        print("Updating metadata with split field...")
        with datasource.metadata_context() as ctx:
            for idx, dp in enumerate(datapoint_list):
                split_value = split_mapping[idx]
                # 获取文件路径作为key
                file_path = dp['path']
                # 更新metadata
                ctx.update_metadata(file_path, {'split': split_value})

        print("Metadata update completed!")

        # 返回统计信息
        stats = {
            'total': total_count,
            'train': train_size,
            'valid': valid_size,
            'test': test_size,
            'train_ratio': train_size / total_count,
            'valid_ratio': valid_size / total_count,
            'test_ratio': test_size / total_count
        }

        return stats

    def create_yolo_v8_dataset_yaml(self, dataset, download=True, label_field="annotation"):
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

    def verify_split_field(self, datasource):
        """
        验证split字段是否已正确添加

        Parameters:
        -----------
        datasource : DagsHub datasource object
            数据源对象

        Returns:
        --------
        dict : 各个split的统计信息
        """
        print("Verifying split field...")

        # 查询各个split的数量
        train_count = len(datasource[datasource['split'] == 'train'].all())
        valid_count = len(datasource[datasource['split'] == 'valid'].all())
        test_count = len(datasource[datasource['split'] == 'test'].all())
        total_count = len(datasource.all())

        # 检查是否有未分配的数据
        has_split = len(datasource[datasource['split'].is_not_null()].all())
        no_split = total_count - has_split

        stats = {
            'total': total_count,
            'train': train_count,
            'valid': valid_count,
            'test': test_count,
            'has_split': has_split,
            'no_split': no_split
        }

        print(f"Total datapoints: {total_count}")
        print(f"Train: {train_count} ({train_count / total_count * 100:.1f}%)")
        print(f"Valid: {valid_count} ({valid_count / total_count * 100:.1f}%)")
        print(f"Test: {test_count} ({test_count / total_count * 100:.1f}%)")
        if no_split > 0:
            print(f"WARNING: {no_split} datapoints without split field!")

        return stats