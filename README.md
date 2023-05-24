# A Tiny Metaverse of Multimodal Natural Language Understanding Classification

The aim of this task is to determine whether a claim in a tweet is worth fact-checking. Typical approaches to make that decision require to either resort to the judgments of professional fact-checkers or to human annotators to answer several auxiliary questions such as "does it contain a verifiable factual claim?", and "is it harmful?", before deciding on the final check-worthiness label.

## Guidelines
The task is offered in Arabic and English. For simplicity, please focus on solving the English contest only!
The scripts of our winning model should be able to run on a single GPU (within 12GB) effortlessly. Both sections are marked while the Section A has 30% and Section B has 70%.
## Section A: Run the winning model (Check-Worthiness)

### Step 0: Prepare the dataset
Download the [dataset](https://mega.nz/file/xWVxAbwa#ibbzHxkkl5A1SBMoMFsirrQ1uYBOOqRmZ-KjT1flQgI), and unzip the structure as follows,
```bash
data
|
en
├── test_data
|    └── images_labeled
|        ├── 925657889473052672.jpg
|        ├── 925746311772692481.jpg
|        ├── 925887908996714497.jpg
|        └── ...
│   ├── CT23_1A_checkworthy_multimodal_english_test_gold.jsonl
│   ├── CT23_1A_checkworthy_multimodal_english_test.jsonl
│   └── features
│       └── test_feats.json
└── train_data
    └── images_labeled
        ├── dev
            └── 1032635895864877056.jpg   
            └── ...           
        ├── dev_test
            └── 1032635895864877056.jpg   
            └── ...   
        ├── merge
            └── 1032635895864877056.jpg   
            └── ...     
        ├── train
            └── 1032635895864877056.jpg   
            └── ...     
    ├── CT23_1A_checkworthy_multimodal_english_dev.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_dev_test.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_merge.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_train.jsonl
    └── features
        ├── dev_feats.json
        ├── dev_test_feats.json
        ├── merge_feats.json
        └── train_feats.json
```
Please use the following method to extract your own features as the features under en are from some old models.
### Step 1: Feature Extraction (All)
```bash
bash scripts/run_feature_extraction_full.sh
```
Put the extracted features and the data into the following structure,
```bash
prompt_ocr_adapter
├── test_data
│   ├── CT23_1A_checkworthy_multimodal_english_test_gold.jsonl
│   ├── CT23_1A_checkworthy_multimodal_english_test.jsonl
│   └── features
│       └── test_feats.json
└── train_data
    ├── CT23_1A_checkworthy_multimodal_english_dev.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_dev_test.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_merge.jsonl
    ├── CT23_1A_checkworthy_multimodal_english_train.jsonl
    └── features
        ├── dev_feats.json
        ├── dev_test_feats.json
        ├── merge_feats.json
        └── train_feats.json
```
### Step 2: Train a Transformer Fusion Layer
Run the following command to train a transformer fusion layer with the english_dev set as the monitor. Record the performance on dev_test.
```python
python blip_feature_extractor.py --train-data-dir ./data/prompt_ocr_adapter \
-d data/prompt_ocr_adapter/train_data \
-s dev \
-tr CT23_1A_checkworthy_multimodal_english_train.jsonl \
-te CT23_1A_checkworthy_multimodal_english_dev.jsonl \
-l en \
--lr 1e-3 \
--train-batch-size 64 \
--heads 12 \
--d 480 \
--model-type adapter \
--num-layers 1
```
With the hyper parameters, now run the following script with all samples and record the performance on the english_test_gold set. Note that the monitor is still the dev.
You are welcome to play around with the hyper parameters but be aware of overfitting the test set is not important as the challege is finished.
```python
python blip_feature_extractor.py --train-data-dir ./data/prompt_ocr_adapter \
-d data/prompt_ocr_adapter/train_data \
-s dev \
-tr CT23_1A_checkworthy_multimodal_english_train.jsonl \
-te CT23_1A_checkworthy_multimodal_english_dev.jsonl \
-l en \
--lr 1e-3 \
--train-batch-size 64 \
--heads 12 \
--d 480 \
--model-type adapter \
--num-layers 1
```
Run the experiments with different random seeds. Well done, if you see similar results as follows,

| Split     | F1          | Accuracy    | Precision  | Recall      |
|-----------|-------------|-------------|------------|-------------|
| dev_test  | 0.7075471698| 0.764945652 | 0.673333333 | 0.8620689655|
| test      | 0.7167902098| 0.775815217 | 0.678343949 | 0.7689530686| 
### Step 3: Train an additional Transformer Fusion Layer with soft prompt removal
Remove the soft prompt of the BLIP2. Repeat the experiments and record the result. What do you observe?
```bash
bash scripts/run_feature_extraction_full.sh
bash scripts/train_transformer_fusion.sh
bash scripts/train_transformer_merge.sh
```
**Hint: Read Blip2Qformer and see what you can do with the attention to query embeds**
### Step 4: Train an additional Transformer Fusion Layer with image removal
Record your results. Do not waste your time on overfitting the test set as your results will be checked.
### Step 5: Train an additional Transformer Fusion Layer with text removal
Likewise, record your results.
### Step 6: Brain storm
Given only 64 examples for two classes, ideate some solutions to solve the problem in Step 2. Specifically, how would you change the model?

Congratulation for finishing Section A!
## Section B: Train a baseline on EmoRegCom_DATA
### Step 0: Prepare the dataset
```
│   test
│   ├── 1_72_0.jpg
│   ├── 2_29_3.jpg
│   ├── ...
│   train
│   ├── 0_3_4.jpg
│   ├── 0_19_3.jpg
│   ├── ...
├── dataset.csv
├── test_dataset.csv
├── test_data.csv
├── train_data.csv
├── train_emotion_labels.csv
├── train_transcriptions.json
└── val_data.csv

```
### Step 1: Preprocessing
To preprocess the data into a format similar to our baseline, use tools such as csv or json reader. **The closer the data is to a check-worthy format**, 
the less effort will be required to customize the dataset. Note that Step 7 requires cross-task validation, so we do not expect the data format
to change a lot.  Please save a copy of your preprocessing script for our format checking test.
**Hint: For OCR, use one of the SOTA models easyocr. Check OCR/easyocr.ipynb and their awesome git [repo](https://github.com/JaidedAI/EasyOCR).**
### Step 2: Feature extraction for EmoRegCom_DATA
mm_feature_extractor.py is a modified feature extractor from blip_feature_extractor.py for a different benchmark dataset. Please modify the script for EmoRegCom_DATA.
### Step 3: Train a Transformer Fusion Layer
Like the step 2 of section A, train transformer fusion layer. Keep the model architecture intact, find a group of useful hyperparameters. Record the average performance on this multimodal NLU dataset and the checkpoints.

**Hint: Check the [Wandb hypersearch notebook](https://mega.nz/file/xDEHXLhY#Wo6swMJ-VG-8QAn98zKFO7iaV1DF5G8fCJWoTuofFUA)**. Also check the sweep_fc.yaml.
### Step 4: Train another additional Transformer Fusion Layer with soft prompt removal
Do not change hyper now. Likewise, record the average performance and the checkpoints.
### Step 5: Train an additional Transformer Fusion Layer with image removal
Likewise, record the average performance and the checkpoints.
### Step 6: Train an additional Transformer Fusion Layer with text removal
Likewise, record the average performance and the checkpoints.
### Step 7: Cross task validation
Test this multimodal model to the features of Section A step 2. Record the results.

Test the model of Section A step 2 on the features of Section B step 3. Record the results.

Please write a brief README file on how to run your script.
```bash
zip -r submission.zip ./PATH_TO_RESULT_AND_SCRIPT 
```
Congratulation for finishing Section B! You just did a great job!
