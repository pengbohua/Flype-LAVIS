#!/bin/bash

for split in train dev_test dev; do
  cat MAMI/train_data/${split}.json >> MAMI/train_data/merge.json
  done

cd MAMI/train_data/images_labeled/merge
ln -s ../dev_test/*.jpg .
ln -s ../dev/*.jpg .
ln -s ../train/*.jpg .