from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


def ce(reduction='mean'):
    return CrossEntropyLoss(reduction=reduction)


def bce(reduction='mean'):
    return BCELoss(reduction=reduction)


def bce_with_logits(reduction='mean'):
    return BCEWithLogitsLoss(reduction=reduction)
