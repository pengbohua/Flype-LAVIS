import argparse
from os.path import dirname
import logging
import json
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from preprocessing.TweetNormalizer import normalizeTweet
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os
from tqdm import tqdm

ROOT_DIR = dirname(dirname(__file__))
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feats(image_dir, text_dir, data_dir, out_file_name, split="train", prompt=True, ocr=True):
    image_names = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    with open(text_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    id_text_dict = {}
    for line in lines:
        tweet = json.loads(line)
        tweet_id = tweet['image_name']
        tweet_text = normalizeTweet(tweet['text_input'])
        # ocr info are in the given data files
        # tweet_ocr = normalizeTweet(tweet['ocr_text'])
        tweet_ocr = None
        if not prompt:
            id_text_dict[tweet_id] = tweet_text + tweet_ocr
        elif not prompt and (not ocr):
            id_text_dict[tweet_id] = tweet_text + tweet_ocr
        # hard prompt
        elif prompt and ocr:
            id_text_dict[tweet_id] = "mnli premise:{}. hypothesis:{}".format(tweet_ocr, tweet_text)
        elif prompt and (not ocr):
            id_text_dict[tweet_id] = "mnli premise: image. hypothesis:{}".format(tweet_text)

    # text extractor and image extractor
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor",
        model_type="pretrain",
        is_eval=True,
        device=device
    )
    print('VL model loaded successfully!')
    out_path = os.path.join(data_dir, 'features')

    image_features = {}
    text_features = {}
    multimodal_features = {}

    for index, image_name in tqdm(enumerate(list(id_text_dict.keys()))):
        image_id = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print("missing file: {}".format(image_path))
            continue
        text = id_text_dict[image_name]

        # process
        image_input = vis_processors["eval"](image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](text)
        sample = {"image": image_input, "text_input": [text_input]}

        # # Multimodal features
        features_multimodal = model.extract_features(sample)
        multimodal_features[image_id] = features_multimodal.multimodal_embeds[:, 0, :].squeeze().flatten().tolist()

        # image features
        features_image = model.extract_features(sample, mode="image")
        image_features[image_id] = features_image.image_embeds_proj[:, 0, :].squeeze().flatten().tolist()

        # # text features
        features_text = model.extract_features(sample, mode="text")
        text_features[image_id] = features_text.text_embeds_proj[:, 0, :].squeeze().flatten().tolist()

    os.makedirs(out_path, exist_ok=True)
    print(len(image_features), len(text_features), len(multimodal_features))
    json.dump({"imgfeats": image_features, "textfeats": text_features, "multifeats": multimodal_features},
              open(os.path.join(out_path, out_file_name), "w"))
    print('Saving split {} successfully!'.format(split))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--train-data-dir", required=False, type=str,
                        default="./MAMI",
                        help="The absolute path to the training data")
    parser.add_argument("--train-out-file-name",  required=False, type=str,
                        default="train_feats.json", help="Output feature file name")

    # dev
    parser.add_argument("--dev-data-dir", required=False, type=str,
                        default="./MAMI",
                        help="The absolute path to the training data")
    parser.add_argument("--dev-out-file-name",  required=False, type=str,
                        default="dev_feats.json", help="Output feature file name")
    # dev_test
    parser.add_argument("--dev-test-data-dir", required=False, type=str,
                        default="./MAMI",
                        help="The absolute path to the training data")
    parser.add_argument("--dev-test-out-file-name",  required=False, type=str,
                        default="dev_test_feats.json", help="Output feature file name")

    # # prompt_adapter
    # parser.add_argument("--merge-data-dir", required=False, type=str,
    #                     default="./data/remove_soft_prompt",
    #                     help="The absolute path to the training data")
    # parser.add_argument("--merge-out-file-name",  required=False, type=str,
    #                     default="merge_feats.json", help="Output feature file name")
    #
    #
    # # test
    # parser.add_argument("--test-data-dir", required=False, type=str,
    #                     default="./data/remove_soft_prompt",
    #                     help="The absolute path to the training data")
    # parser.add_argument("--test-out-file-name",  required=False, type=str,
    #                     default="test_feats.json", help="Output feature file name")

    args = parser.parse_args()

    ct23_train_image_dir = 'MAMI/train'
    ct23_dev_image_dir = 'MAMI/train'
    ct23_dev_test_image_dir = 'MAMI/train'
    # ct23_test_image_dir = './data/en/test_data/images_labeled/test/'
    # ct23_merge_image_dir = './data/en/train_data/images_labeled/merge/'

    ct23_train_text_dir = 'MAMI/train.json'
    ct23_dev_text_dir = 'MAMI/dev.json'
    ct23_dev_test_text_dir = 'MAMI/test.json'
    # ct23_test_text_dir = './data/en/test_data/CT23_1A_checkworthy_multimodal_english_test.jsonl'
    # ct23_merge_text_dir = './data/en/train_data/CT23_1A_checkworthy_multimodal_english_merge.jsonl'

    get_feats(ct23_dev_image_dir, ct23_dev_text_dir, args.dev_data_dir, args.dev_out_file_name, prompt=True, ocr=False)
    get_feats(ct23_dev_test_image_dir, ct23_dev_test_text_dir, args.dev_test_data_dir, args.dev_test_out_file_name, prompt=True, ocr=False)
    # get_feats(ct23_test_image_dir, ct23_test_text_dir, args.test_data_dir, args.test_out_file_name, prompt=True, ocr=True)

    get_feats(ct23_train_image_dir, ct23_train_text_dir, args.train_data_dir, args.train_out_file_name, prompt=True, ocr=False)
    # get_feats(ct23_merge_image_dir, ct23_merge_text_dir, args.merge_data_dir, args.merge_out_file_name, prompt=True, ocr=True)

