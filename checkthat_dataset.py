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
            elif label[1] == 'No':
                converted_labels.append(0)
            else:
                raise ValueError
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


