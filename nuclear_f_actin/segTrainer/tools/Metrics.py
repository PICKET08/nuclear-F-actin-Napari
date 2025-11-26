import numpy as np
from scipy import spatial


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def recall(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    recall = np.divide(TP, TP + FN)
    return recall * 100.0

def precision(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    precision = np.divide(TP, TP + FP)
    return precision * 100.0

def dice_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N1 = 2 * TP
    N2 = 2 * TP + FP + FN
    dice = np.divide(N1, N2)
    return dice * 100.0

def IoU(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    IoU = np.divide(TP, TP + FP + FN)
    return IoU * 100.0