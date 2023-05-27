import easyocr
import os
import json
import numpy as np
import argparse
import cv2
import threading
from queue import Queue
import time


def _save_image_in_queue(th_id, img_paths, img_queue):
    for idx in range(len(img_paths)):
        img_path = img_paths[idx]
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        if img is not None:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_queue.put({'image_name': img_name, 'image': gray_image})
    print("Thread {} processing ...".format(th_id))


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
def _thread_safe_read_image(img_dict):
    _res = ocr_reader.readtext(img_dict['image'])
    return {'image_name': img_dict['image_name'], 'ocr_res': _res}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='text to image reader')
    parser.add_argument('--input-path', type=str, required=False, default='MAMI/test_data/images_labeled/test', help='input directory for images')
    parser.add_argument('--output-path', type=str, required=False, default='MAMI/test_data/', help='output directory for images')
    parser.add_argument('--queue-size', type=int, required=False, default=10000, help='number of images in RAM')
    parser.add_argument('--num-threads', type=int, required=False, default=5, help='number of threads used for ocr')

    args = parser.parse_args()

    image_queue = Queue(args.queue_size)
    ocr_reader = easyocr.Reader(['en'])
    image_paths = find_jpg_paths(args.input_path)
    total_results = []
    threads = []
    time1 = time.time()
    # parallel input processing
    if len(image_paths) > args.queue_size:
        print("Warning: the number of images is big, processing only first {} images.".format(len(args.queue_size)))
    bs = len(image_paths) // args.num_threads + 1
    _id = 0
    for i in range(0, len(image_paths), bs):
        image_list = image_paths[i: i+bs]
        thread = threading.Thread(target=_save_image_in_queue, args=(_id, image_list, image_queue))
        threads.append(thread)
        thread.start()
        _id += 1

    for thread in threads:
        thread.join()
    print("queue size", image_queue.qsize())
    images_results = [_thread_safe_read_image(image_queue.get()) for _ in range(image_queue.qsize())]
    total_results.extend(images_results)

    with open("MAMI/test_data/t_ocr_results.json", "w") as f:
        json.dump(total_results, f, cls=NumpyJSONEncoder, indent=4)
