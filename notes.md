# Task 1: Check-Worthiness in Multimodal and Unimodal Content

repo: git@github.com:pengbohua/CheckthatMM2023.git https://github.com/pengbohua/CheckthatMM2023.git
## Run the code
先特征抽取
python extract_feat_1a.py -d  data/ -f CT23_1A_checkworthy_multimodal_english_train.jsonl -o train_feats.json -l en
python extract_feat_1a.py -d data/ -f CT23_1A_checkworthy_multimodal_english_dev.jsonl -o dev_feats.json -l en

python main.py -d data/train_data/ -s dev -tr CT23_1A_checkworthy_multimodal_english_train.jsonl -te CT23_1A_checkworthy_multimodal_english_dev.jsonl -l en
python main.py -d data/test_data/ -s test -tr CT23_1A_checkworthy_multimodal_english_train.jsonl -te CT23_1A_checkworthy_multimodal_english_test.jsonl -l english

## Tokenizer
tokenizer过滤url, userid TweetNormalizer

## modelling
naive_fuser

## CustomTrainer


## train
line68
```python
    trainer.train()
    print("Saving checkpoint in checkpoints/")
    trainer.save_model("checkpoints/0.pt")
```
## Predictor / Test
```python
    test_model(data_dir, test_split, test_fpath, imgbert_baseline_fpath)
```
## format check before submission
python3 format_checker/subtask_1.py --pred-files-path results/imgbert_baseline_CT23_1A_checkworthy_multimodal_english_test.txt

## DataAnalysis
明天干
