import torch
from transformers import Trainer
import torch.nn.functional as F
import torch.nn as nn
import torch


class CustomTrainer(Trainer):
    def __init__(self, alpha=0.25, gamma=2.0, model_type="fc", **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.model_type = model_type
        self.criterion = nn.BCEWithLogitsLoss(weight=torch.Tensor([2.0])).cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the compute_loss() method to implement Focal Loss.
        """
        labels = inputs.pop("labels")
        if self.model_type == 'adapter':
            multifeats = inputs.pop("multi_tensor")
        elif self.model_type == 'fc':
            image_feats = inputs.pop("img_tensor")
            text_feats = inputs.pop("text_tensor")
        else:
            raise NotImplementedError
        outputs = model(**inputs)
        loss = self.criterion(outputs['logits'].squeeze(), labels.float())
        return (loss, outputs) if return_outputs else loss        # # Compute the class weights based on the alpha parameter
        # class_weights = torch.ones(logits.shape[1])
        # class_weights[labels == 1] = 1 - self.alpha
        # class_weights[labels == 0] = self.alpha
        #
        # # Apply Focal Loss for imbalance data (Kaiming)
        # pt = torch.exp(-F.binary_cross_entropy_with_logits(logits, labels, reduction='none'))
        # focal_loss = (class_weights * (1 - pt) ** self.gamma * F.binary_cross_entropy_with_logits(logits, labels,
        #                                                                                           reduction='none')).mean()
        #
        # return (focal_loss, logits) if return_outputs else focal_loss


