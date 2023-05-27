from transformers import Trainer
import torch.nn.functional as F
import torch.nn as nn
import torch


#TODO implement multiclass classification


class FocalLoss(nn.Module):
    def __init__(self, weight=1.0, gamma=0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = torch.Tensor([weight]).cuda()

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        bce = F.binary_cross_entropy(probs, targets.float(), weight=self.weight, reduction='none')
        factor = (1 - probs).pow(self.gamma)
        focal_loss = factor * bce
        return focal_loss.mean()


class CustomTrainer(Trainer):
    def __init__(self, weight=1.0, gamma=2.0, model_type="fc", loss_type='wbce', **kwargs):
        super().__init__(**kwargs)
        self.weight = torch.Tensor([weight])
        self.gamma = gamma
        self.model_type = model_type
        self.loss_type = loss_type
        if loss_type == 'wbce':
            self.criterion = nn.BCEWithLogitsLoss(weight=self.weight).cuda()
        elif loss_type == 'bfocal':
            self.criterion = FocalLoss().cuda()
        else:
            raise NotImplementedError

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
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    # sanity check
    criterion = FocalLoss()
    inputs = torch.randn(1)
    labels = torch.LongTensor([1])
    print(criterion(inputs, labels))