#!/bin/bash

python main.py --data-dir ./data/prompt_ocr_adapter/train_data \
-s dev \
-tr CT23_1A_checkworthy_multimodal_english_merge.jsonl \
-te CT23_1A_checkworthy_multimodal_english_dev.jsonl \
-l en \
--lr 1e-3 \
--train-batch-size 64 \
--heads 12 \
--d 480 \
--model-type adapter \
--num-layers 1