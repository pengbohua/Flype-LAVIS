import torch
from transformers import Trainer, TrainingArguments, IntervalStrategy
import torch.nn as nn


training_args = TrainingArguments(
    evaluation_strategy=IntervalStrategy.STEPS,  # "steps"
    eval_steps=50,  # Evaluation and Save happens every 50 steps
    output_dir='./checkpoints',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    dataloader_drop_last=False,
    logging_steps=10,
    metric_for_best_model='accuracy',
    load_best_model_at_end=True
)


def compute_focal_loss(self, model, inputs, return_outputs=False, alpha=0.25, gamma=2.0):
    """
    Override the compute_loss() method to implement Focal Loss.
    """
    labels = inputs.pop("labels")
    logits = model(**inputs)
    loss = self.criterion(logits, labels)
    return loss
    # Compute the class weights based on the alpha parameter
    class_weights = torch.ones(logits.shape[1])
    class_weights[labels == 1] = 1 - self.alpha
    class_weights[labels == 0] = self.alpha

    # Apply Focal Loss for imbalance data (Kaiming)
    pt = torch.exp(-F.binary_cross_entropy_with_logits(logits, labels, reduction='none'))
    focal_loss = (class_weights * (1 - pt) ** self.gamma * F.binary_cross_entropy_with_logits(logits, labels,
                                                                                              reduction='none')).mean()

    return (focal_loss, logits) if return_outputs else focal_loss


def compute_loss(model, inputs):
    labels = inputs.pop('labels')
    outputs = model(**inputs)
    logits = outputs.logits
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss