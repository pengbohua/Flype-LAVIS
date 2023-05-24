import torch
from torch.utils.data import Dataset
from lavis.common.registry import registry
from PIL import Image
import jsonlines

class MMDataset(Dataset):
    def __init__(self, image_data, text_data, multimodal_data, labels):
        self.image_data = image_data
        self.text_data = text_data
        self.multimodal_data = multimodal_data
        self.tweet_ids = []
        self.labels = self.convert_labels(labels)
        print("Loaded {} examples".format(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_data[index]
        text = self.text_data[index]
        label = self.labels[index]
        multimodal = self.multimodal_data[index]
        return image, text, multimodal, label

    def convert_labels(self, labels):
        converted_labels = []
        for label in labels:
            if label[1] == 'Yes':
                converted_labels.append(1)
            else:
                converted_labels.append(0)
            self.tweet_ids.append(label[0])
        return torch.LongTensor(converted_labels)


class MMTestDataset(Dataset):
    def __init__(self, image_data, text_data, multimodal_data, tweet_ids):
        self.image_data = image_data
        self.text_data = text_data
        self.multimodal_data = multimodal_data
        self.tweet_ids = tweet_ids
        print("Loaded {} test examples".format(len(text_data)))

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        image = self.image_data[index]
        text = self.text_data[index]
        multimodal = self.multimodal_data[index]
        return image, text, multimodal, None


class BLIPTrainDataset(Dataset):
    def __init__(self, image_data, text_data, labels):
        from lavis.common.registry import registry
        from lavis.models import load_model_and_preprocess

        # BLIP_ImageTrain = registry.get_processor_class('blip_image_train')  # model class
        # BLIP_Text = registry.get_processor_class('blip_question')

        _, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device='cpu'
        )
        self.image_preprocess = vis_processors['train']
        self.text_preprocess = txt_processors['train']

        self.image_data_list = image_data
        self.text_data_list = text_data
        self.tweet_ids = []
        self.labels = self.convert_labels(labels)
        print("Loaded {} examples".format(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_data_list[index]
        self.image_preprocess(Image.open(image))
        text = self.text_data_list[index]
        self.text_preprocess(text)
        label = self.labels[index]
        return image, text, label

    def convert_labels(self, labels):
        converted_labels = []
        for label in labels:
            if label[1] == 'Yes':
                converted_labels.append(1)
            else:
                converted_labels.append(0)
            self.tweet_ids.append(label[0])
        return torch.LongTensor(converted_labels)


class BLIPTestDataset(Dataset):
    def __init__(self, image_data, text_data, labels):

        from lavis.models import load_model_and_preprocess

        _, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device='cpu'
        )
        self.image_preprocess = vis_processors['eval']
        self.text_preprocess = txt_processors['eval']
        self.image_data_list = image_data
        self.text_data_list = text_data
        self.tweet_ids = []
        self.labels = self.convert_labels(labels)
        print("Loaded {} examples".format(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_data_list[index]
        self.image_preprocess(image)
        text = self.text_data_list[index]
        self.text_preprocess(text)
        label = self.labels[index]
        return image, text, label

    def convert_labels(self, labels):
        converted_labels = []
        for label in labels:
            if label[1] == 'Yes':
                converted_labels.append(1)
            else:
                converted_labels.append(0)
            self.tweet_ids.append(label[0])
        return torch.LongTensor(converted_labels)


def collate_func(batch_data):
    images = []
    texts = []
    multifeats = []
    labels = []
    for _b in batch_data:
        images.append(_b[0])
        texts.append(_b[1])
        multifeats.append(_b[2])
        labels.append(_b[3])

    return {"img_tensor": torch.stack(images, 0),
            "text_tensor": torch.stack(texts, 0),
            "multi_tensor": torch.stack(multifeats, 0),
            "labels": torch.stack(labels, 0) if _b[3] is not None else None
            }


def blip_collate_func(batch_data):
    images = []
    texts = []
    labels = []
    for _b in batch_data:
        images.append(_b[0])
        texts.append(_b[1])
        labels.append(_b[3])

    return {"img_tensor": torch.stack(images, 0),
            "text_tensor": torch.stack(texts, 0),
            "labels": torch.stack(labels, 0) if _b[2] is not None else None
            }


if __name__ == '__main__':
    from os.path import join, dirname, basename
    from torch.utils.data import DataLoader
    data_dir = 'data/ocr_en/train_data'
    img_dir = 'data/en/train_data'
    train_fpath = 'CT23_1A_checkworthy_multimodal_english_train.jsonl'
    total_tweet_ids = []
    total_class_labels = []
    total_texts = []
    total_image_paths = []
    for obj in jsonlines.open(join(data_dir, train_fpath)):
        tweet_id = obj["tweet_id"]
        class_label = obj["class_label"]
        text = obj["tweet_text"]
        image_path = obj["image_path"]

        total_tweet_ids.append(tweet_id)
        total_class_labels.append(class_label)
        total_texts.append(text)
        total_image_paths.append(join(img_dir, image_path))
    blip_train_dataset = BLIPTrainDataset(total_image_paths, total_texts, total_class_labels)

    from lavis.models import load_model_and_preprocess
    from modeling.qformer import Blip2Qformer

    train_loader = DataLoader(blip_train_dataset, batch_size=16, num_workers=1, collate_fn=blip_collate_func)
    data = next(iter(train_loader))
    print(data)
    model, _, _ = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain",
    is_eval=True,
    device='cpu'
)
    model = Blip2Qformer()