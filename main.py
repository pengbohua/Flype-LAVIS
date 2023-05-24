import jsonlines
import json
import random
import logging
import argparse
from os.path import join, dirname, basename
import numpy as np
from modeling.naive_fuser import MMFusionTransformer
from modeling.mlp_linear_classifier import MLPClassifier
import torch
import os
import sys
sys.path.append('preprocessing')
from scorer.subtask_1 import evaluate
from format_checker.subtask_1 import check_format
from custom_dataset import MMDataset, MMTestDataset, collate_func
from torch.utils.data import DataLoader
from trainer.trainer import CustomTrainer
import wandb
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments, IntervalStrategy
from sklearn.metrics import accuracy_score


random.seed(1234)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def compute_metrics(p):
    pred, labels = p
    pred = np.greater(pred, 0.25).squeeze()
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    wandb.log({"val accuracy": accuracy})
    return {"accuracy": accuracy,}


def train_qformer(data_dir, split, train_fpath, test_fpath, args):
    """

    @param data_dir:
    @param split:
    @param train_fpath:
    @param test_fpath:
    @param results_fpath: results/
    @param model_id:
    """

    training_args = TrainingArguments(
        evaluation_strategy=IntervalStrategy.STEPS,  # "steps"
        eval_steps=100,  # Evaluation and Save happens every 50 steps
        output_dir='./checkpoints',
        num_train_epochs=5,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True
    )

    tr_feats = json.load(open(join(data_dir, "features", "merge_feats.json")))  # format {"imgfeats":{"tweetid":768d}}
    te_feats = json.load(open(join(data_dir, "features", "%s_feats.json"%(split))))
    _train_id_labels = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, train_fpath))]    # format: [["tweetid": "Yes"]]
    if "dev" in test_fpath:
        test_id_labels = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, test_fpath))]
    else:
        raise ValueError("dev not in validation set name")

    tr_img_feats = []
    tr_text_feats = []
    tr_multi_feats = []
    train_id_labels = []
    for obj in _train_id_labels:
        try:
            tr_img_feat = tr_feats["imgfeats"][obj[0]]
            tr_text_feat = tr_feats["textfeats"][obj[0]]
            tr_multi_feat = tr_feats["multifeats"][obj[0]]
        except KeyError:
            continue
        else:
            tr_multi_feats.append(tr_multi_feat)
            tr_img_feats.append(tr_img_feat)
            tr_text_feats.append(tr_text_feat)
            train_id_labels.append(obj)

    tr_img_feats = np.array(tr_img_feats)
    tr_text_feats = np.array(tr_text_feats)
    tr_multi_feats = np.array(tr_multi_feats)

    tr_img_feat = torch.from_numpy(tr_img_feats).float()
    # tr_img_feat = torch.randn_like(tr_img_feat)     # replace image with random noise for ablation study
    tr_text_feat = torch.from_numpy(tr_text_feats).float()
    # tr_text_feat = torch.randn_like(tr_text_feat) # replace text with random noise for ablation study
    tr_multi_feat = torch.from_numpy(tr_multi_feats).float()

    train_dataset = MMDataset(tr_img_feat, tr_text_feat, tr_multi_feat, train_id_labels)
    # 820 / 2356
    # save valid outputs
    val_img_feats = [te_feats["imgfeats"][obj[0]] for obj in test_id_labels]
    val_img_feats = np.array(val_img_feats)
    val_img_feats = torch.from_numpy(val_img_feats).float()
    val_text_feats = [te_feats["textfeats"][obj[0]] for obj in test_id_labels]
    val_text_feats = np.array(val_text_feats)
    val_text_feats = torch.from_numpy(val_text_feats).float()
    val_multi_feats = [te_feats["multifeats"][obj[0]] for obj in test_id_labels]
    val_multi_feats = np.array(val_multi_feats)
    val_multi_feats = torch.from_numpy(val_multi_feats).float()
    valid_dataset = MMDataset(val_img_feats, val_text_feats, val_multi_feats, test_id_labels)
    # val: 87 / 271; 174 / 548
    if args.model_type == 'adapter':
        model = MMFusionTransformer(n_heads=args.heads, hidden_dim=args.d, dropout=0.1, num_layers=args.num_layers)
    elif args.model_type == 'fc':
        model = MLPClassifier(hidden_dim=args.d)
    else:
        raise NotImplementedError

    model = model.cuda()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_func,
        compute_metrics=compute_metrics,
        model_type=args.model_type,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02, )],
    )
    trainer.train()
    print("Saving checkpoint in checkpoints/")
    os.makedirs("checkpoints/mlp_ocr", exist_ok=True)
    trainer.save_model("checkpoints/mlp_ocr/bs_{}_lr{}_heads{}_d{}.pt".format(args.train_batch_size, args.lr, args.heads, args.d))


def test_model(data_dir, split, test_fpath, results_fpath, lang, model_id='imagebert', args=None):

    te_feats = json.load(open(join(data_dir, "features", "%s_feats.json"%(split))))

    test_id_labels = [obj["tweet_id"] for obj in jsonlines.open(join(data_dir, test_fpath))]
    te_img_feats = [te_feats["imgfeats"][obj] for obj in test_id_labels]
    te_img_feats = np.array(te_img_feats)
    te_text_feats = [te_feats["textfeats"][obj] for obj in test_id_labels]
    te_text_feats = np.array(te_text_feats)
    te_multi_feats = [te_feats["multifeats"][obj] for obj in test_id_labels]
    te_multi_feats = np.array(te_multi_feats)

    te_img_feats = torch.from_numpy(te_img_feats).float().cuda()
    te_text_feats = torch.from_numpy(te_text_feats).float().cuda()
    te_multi_feats = torch.from_numpy(te_multi_feats).float().cuda()
    test_dataset = MMTestDataset(te_img_feats, te_text_feats, te_multi_feats, test_id_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)
    if args.model_type == 'adapter':
        model = MMFusionTransformer(n_heads=args.heads, hidden_dim=args.d, dropout=0.1, num_layers=args.num_layers).cuda()
    elif args.model_type == 'fc':
        model = MLPClassifier(hidden_dim=args.d)
    else:
        raise NotImplementedError
    print("Load checkpoints from checkpoints/mlp_ocr/bs_{}_lr{}_heads{}_d{}.pt/pytorch_model.bin".format(args.train_batch_size, args.lr, args.heads, args.d))
    state_dict = torch.load("checkpoints/mlp_ocr/bs_{}_lr{}_heads{}_d{}.pt/pytorch_model.bin".format(args.train_batch_size, args.lr, args.heads, args.d))
    model.load_state_dict(state_dict)
    model = model.cuda()
    ## Write test file in format
    with open(results_fpath, "w") as results_file:
        results_file.write("tweet_id\tclass_label\trun_id\n")

        for i, batch_dict in enumerate(test_loader):
            tweet_id = test_dataset.tweet_ids[i]
            batch_dict.pop("labels")
            if args.model_type == 'adapter':
                multifeats = batch_dict.pop("multi_tensor")
            elif args.model_type == 'fc':
                image_feats = batch_dict.pop("img_tensor")
                text_feats = batch_dict.pop("text_tensor")
            else:
                raise NotImplementedError
            outputs = model(**batch_dict)
            prd = torch.sigmoid(outputs['logits']).item()
            if prd > 0.25:
                label = "Yes"
            else:
                label = "No"

            results_file.write("{}\t{}\t{}\n".format(tweet_id, label, "{}".format(model_id)))

    gold_fpath = join(data_dir, f'{basename(test_fpath)}')

    # evaluation on dev
    if check_format(results_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, results_fpath, subtask="A")
        logging.info(f"Qformer for {lang} Accuracy (positive class): {acc}")
        logging.info(f"Qformer for {lang} Precision (positive class): {precision}")
        logging.info(f"Qformer for {lang} Recall (positive class): {recall}")
        logging.info(f"Qformer for {lang} F1 (positive class): {f1}")
    with open("results/hypersearch.txt", "a") as f:
        f.write("bs_{}_lr{}_heads{}_d{}\t {}".format(args.train_batch_size, args.lr, args.heads, args.d, f1))


def supervised_training(data_dir, test_split, train_fpath, test_fpath, lang, args):
    # run training function
    train_qformer(data_dir, test_split, train_fpath, test_fpath, args)
    # run test function on dev_test
    test_model(data_dir="data/ocr_en/train_data/", split="dev_test", test_fpath='CT23_1A_checkworthy_multimodal_english_dev_test.jsonl',
               results_fpath="results/mlp_ocr_en_subtask1A_en_dev_test.tsv".format(args.train_batch_size, args.lr, args.heads, args.d), lang=lang, args=args, model_id="baseline_adapter")
    # run test function on test
    test_model(data_dir="data/ocr_en/test_data/", split="test", test_fpath='CT23_1A_checkworthy_multimodal_english_test_gold.jsonl',
               results_fpath="results/mlp_ocr_en_subtask1A_en_test.tsv".format(args.train_batch_size, args.lr, args.heads, args.d), lang=lang, args=args, model_id="baseline_adapter")


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=False, type=str,
                        default="data/ocr_en/train_data",
                        help="The absolute path to the training data")
    parser.add_argument("--test-split", "-s", required=False, type=str,
                        default="dev", help="Test split name")
    parser.add_argument("--train-file-name", "-tr", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_train.jsonl",
                        help="Training file name")
    parser.add_argument("--test-file-name", "-te", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_dev.jsonl",
                        help="Test file name")
    parser.add_argument("--lang", "-l", required=False, type=str, default="english",
                        help="Options: arabic | english")
    parser.add_argument("--lr", required=False, type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--train-batch-size", required=False, type=int, default=64,
                        help="training batch size")
    parser.add_argument("--heads", required=False, type=int, default=12,
                        help="heads")
    parser.add_argument("--d", required=False, type=int, default=480,
                        help="hidden_dimension")
    parser.add_argument("--num-layers", required=False, type=int, default=1,
                        help="hidden_dimension")
    parser.add_argument("--model-type", required=False, type=str, default="adapter",
                        help="fc or adapter")
    args = parser.parse_args()

    # args.heads = config.heads
    # args.d = config.d
    # args.lr = config.lr
    # args.train_batch_size = config.train_batch_size
    # args.num_layers = config.num_layers
    wandb.init(mode="disabled",)
    # wandb.init(entity='marvinpeng', project="checkthat",) # replace the entity with your name and your project to run a hyper parameter sweep
    supervised_training(args.data_dir, args.test_split, args.train_file_name, args.test_file_name, args.lang, args)


if __name__ == '__main__':
    main()