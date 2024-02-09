import numpy as np
import keras
import keras.backend as K


def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def DiceBCELoss(targets, inputs, smooth=1e-6):
    # Combines Dice with BCE
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE =  K.binary_crossentropy(targets, inputs)
    intersection = K.sum(targets * inputs)
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE


def IoULoss(targets, inputs, smooth=1e-6):
    # Jaccard index
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def FocalLoss(targets, inputs, alpha=1, gamma=1):
    # includes gamma modifier that deviates the loss from cross entropy
    # alpha modifier
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss


def TverskyLoss(targets, inputs, alpha=0.5, beta=0.7, smooth=1e-6):
    # alpha controls the penalty for false positives
    # beta controls the penalty for false negatives
    # alpha = beta = 0.5 => Dice loss
    # alpha = beta = 1 => Tanimoto coefficient
    # alpha + beta = 1 => set of Fbeta scores
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
    return 1 - Tversky


def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.7, gamma=1, smooth=1e-6):
    # Variant of Tversky that includes gamma modifier
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
            
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1 - Tversky), gamma)
    
    return FocalTversky
    

def ComboLoss(targets, inputs, alpha=0.6, ce_ratio=0.5, eps=1e-9):
    # alpha < 0.5 penalises FP more, > 0.5 penalises FN more
    # ce_ratio weighted contribution of modified CE loss compared to Dice loss
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + eps) / (K.sum(targets) + K.sum(inputs) + eps)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (alpha * ((targets * K.log(inputs)) + ((1 - alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)
    
    return combo
