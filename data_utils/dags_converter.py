       
import json

from label_studio_converter.converter import Converter

class DagsConverter(Converter):
    def iter_from_json_file(self, json_file):
        """Extract annotation results from json file

        param json_file: path to task dict with annotations
        """
        with open(json_file) as j: 
            data = json.load(j)
            if not id in data:
                data['id'] = 0
            for item in self.annotation_result_from_task(data):
                yield item
            
    def _get_labels(self):
        labels = set()
        categories = list()
        category_name_to_id = dict()
        for name, info in self._schema.items():
            labels |= set(info['labels'])
            attrs = info['labels_attrs']
            for label in attrs:
                # DagsHub: add handling for case where there is no category attribute but we want to use the order in the schema
                label_num = attrs[label].get('category') if attrs[label].get('category') else info['labels'].index(label)
                categories.append(
                    {'id': label_num, 'name': label}
                )
                category_name_to_id[label] = label_num
        labels_to_add = set(labels) - set(list(category_name_to_id.keys()))
        labels_to_add = sorted(list(labels_to_add))
        idx = 0
        while idx in list(category_name_to_id.values()):
            idx += 1
        for label in labels_to_add:
            categories.append({'id': idx, 'name': label})
            category_name_to_id[label] = idx
            idx += 1
            while idx in list(category_name_to_id.values()):
                idx += 1
        return categories, category_name_to_id