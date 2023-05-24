#!/bin/bash

python blip_feature_extractor.py --train-data-dir ./data/prompt_ocr_adapter \
--train-out-file-name train_feats.json \
--dev-data-dir ./data/prompt_ocr_adapter \
--dev-out-file-name dev_feats.json \
--dev-test-data-dir ./data/prompt_ocr_adapter \
--dev-test-out-file-name dev_test_feats.json \
--merge-data-dir ./data/prompt_ocr_adapter \
--merge-out-file-name merge_feats.json \
--test-data-dir ./data/prompt_ocr_adapter \
--test-out-file-name test_feats.json
