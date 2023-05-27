import easyocr
import os
import json
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

def find_jpg_paths(directory):
    jpg_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_path = os.path.join(root, file)
                jpg_paths.append(jpg_path)
    return jpg_paths


def read_image(paths, reader):
    ocr_results = []
    for p in paths:
        _ocr = {}
        img_name = p.split('/')[-1]
        res = reader.readtext(p, slope_ths=0.2)
        _ocr['image_name'] = img_name
        _ocr['ocr_res'] = res
        ocr_results.append(_ocr)
    return ocr_results


if __name__ == '__main__':
    ocr_reader = easyocr.Reader(['en'])
    jpg_paths = find_jpg_paths('MAMI/train_data/images_labeled/merge')
    results = read_image(jpg_paths[:2], ocr_reader)
    with open("MAMI/train_data/ocr_results.json", "w") as f:
        json.dump(results, f, cls=NumpyJSONEncoder)
