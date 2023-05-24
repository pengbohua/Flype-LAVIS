#!/bin/bash

for split in train dev_test dev; do
  cat data/en/train_data/CT23_1A_checkworthy_multimodal_english_${split}.jsonl >> data/en/train_data/CT23_1A_checkworthy_multimodal_english_merge.jsonl
  done

cd data/en/train_data/images_labeled/merge
ln -s ../dev_test/*.jpg .
ln -s ../dev/*.jpg .
ln -s ../train/*.jpg .