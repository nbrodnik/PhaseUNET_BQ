
import keras.backend as K


def Recall(y_true, y_pred, smooth=1e-6):
    true_positives = (y_true * y_pred).sum()
    possible_positives = y_true.sum()
    recall = true_positives / (possible_positives + smooth)
    return recall


def Precision(y_true, y_pred, smooth=1e-6):
    true_positives = (y_true * y_pred).sum()
    predicted_positives = (y_pred).sum()
    precision = true_positives / (predicted_positives + smooth)
    return precision


def F1(y_true, y_pred, smooth=1e-6):
    # Also called DSC
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return 2*((precision*recall)/(precision+recall+smooth))


def Jaccard_coef(y_true, y_pred, smooth=1):
    # Also called IoU
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    intersecion = (y_true * y_pred).sum()
    union = (y_true + y_pred).sum() - intersecion
    jac = (intersecion + smooth) / (union + smooth)
    return jac

def Jaccard_coef_Keras(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac