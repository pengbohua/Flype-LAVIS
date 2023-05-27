# -*- coding: utf-8 -*-
# @Time    : 2023/5/23 7:18
# @Author  : yujin wang
# @Email   : yujeen.wang@gmail.com


import os
import jsonlines
from tqdm import tqdm
import json
import argparse
import sys
from os.path import dirname
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from TweetNormalizer import normalizeTweet
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = dirname(dirname(__file__))
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Blip2MMDataset(Dataset):
    def __init__(self, jsonl_file, data_dir):
        self.jsonl_file = [obj for obj in jsonlines.open(os.path.join(data_dir, jsonl_file))]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.jsonl_file)

    def __getitem__(self, idx):
        obj = self.jsonl_file[idx]
        tweet_id = obj["tweet_id"]
        image_path = os.path.join(self.data_dir, obj["image_path"])
        text = normalizeTweet(obj["tweet_text"])
        label = obj["class_label"]
        return tweet_id, image_path, text, label

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, model_blip2, model_mlp):
        super(CombinedModel, self).__init__()
        self.mlp = model_mlp
        self.blip2 = model_blip2

    def forward(self, **inputs):
        qformer_features = self.blip2(**inputs).qformer_outputs.last_hidden_state[:, 0, :].squeeze()
        x = self.mlp(qformer_features)
        return x


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=False, type=str,
                        default="../data/CT23_1A_checkworthy_multimodal_english_v2/",
                        help="The absolute path to the training data")

    parser.add_argument("--file-name-train", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_train.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--file-name-val", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_dev.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--file-name-test", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_test.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--out-file-name", "-o", required=False, type=str,
                        default="train_feats.json", help="Output feature file name")
    parser.add_argument("--lang", "-l", required=False, type=str, default="en",
                        help="Options: ar | en")
    args = parser.parse_args()

    # model load
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model_blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    model_mlp = MLP(input_size=768, hidden_size=256, output_size=2).to(device, dtype=torch.float16)
    combined_model = CombinedModel(model_blip2=model_blip2, model_mlp=model_mlp)
    del model_blip2
    del model_mlp

    # Training settings
    data_dir = args.data_dir
    train_file = args.file_name_train
    val_file = args.file_name_val
    test_file = args.file_name_test
    out_path = os.path.join(data_dir, "features")
    batch_size = 2
    learning_rate = 3e-4
    epochs = 10

    # Dataset and DataLoader
    train_dataset = Blip2MMDataset(train_file, data_dir)
    val_dataset = Blip2MMDataset(val_file, data_dir)
    test_dataset = Blip2MMDataset(test_file,data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        combined_model.train()
        total_loss = 0.0
        predictions = []
        targets = []

        for batch_index, (tweet_id, image_path, text, label) in enumerate(tqdm(train_loader)):
            image_list = [Image.open(img_pth).convert("RGB") for img_pth in image_path]
            text_list = [txt for txt in text]
            label_list = torch.tensor([1 if lab == "Yes" else 0 for lab in label]).to(device)
            inputs = processor(
                images=[img for img in image_list],
                text=[txt for txt in text_list],
                return_tensors="pt",
                padding=True
            ).to(device, torch.float16)

            optimizer.zero_grad()
            logits = combined_model(**inputs)
            loss = criterion(logits, label_list)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            targets.extend(label_list.tolist())

        epoch_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

        # Validation loop
        combined_model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for val_batch_index, (val_tweet_id, val_image_path, val_text, val_label) in enumerate(tqdm(val_loader)):
                val_image_list = [Image.open(img_pth).convert("RGB") for img_pth in val_image_path]
                val_text_list = [txt for txt in val_text]
                val_label_list = torch.tensor([1 if lab == "Yes" else 0 for lab in val_label]).to(device)

                val_inputs = processor(
                    images=[img for img in val_image_list],
                    text=[txt for txt in val_text_list],
                    return_tensors="pt",
                    padding=True
                ).to(device, torch.float16)

                val_logits = combined_model(**val_inputs)
                val_predictions.extend(torch.argmax(val_logits, dim=1).tolist())
                val_targets.extend(val_label_list.tolist())

            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_f1 = f1_score(val_targets, val_predictions)

            print(f"Validation Accuracy: {val_accuracy:.4f} | Validation F1 Score: {val_f1:.4f}")

        torch.save(combined_model.state_dict(), "trained_model.pth")


