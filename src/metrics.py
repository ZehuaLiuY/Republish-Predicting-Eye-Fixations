import numpy as np
from scipy.integrate import simps
import random

from scipy.ndimage import zoom


def roc_auc(pred, target, n_points=20, include_prior=False):
    """
        Calculates the Reciever-Operating-Characteristic (ROC) area under
        the curve (AUC) by numerical integration.
        """

    target = np.array(target) / 255
    generated = pred
    # min max normalisation
    generated = (generated - generated.min()) / (generated.max() - generated.min())

    def roc(p=0.1):
        x = generated.reshape(-1) > p
        t = target.reshape(-1) > p

        return np.sum(x == t) / len(t)

    calculate_roc = np.vectorize(roc)

    x = np.linspace(0, 1, n_points)
    auc = simps(calculate_roc(x)) / n_points

    return auc


def calculate_auc(preds, targets):
    """
        inputs -- 2 dictionary with prediction and target images. The 2 dictionaries have the  same number of keys,
        where each key identifies an unique image.
        The predictions have the predicted fixation maps while the targets have the ground truth fixation maps
        which are available from "https://people.csail.mit.edu/tjudd/WherePeopleLook/"
        """
    assert preds.keys() == targets.keys()
    mean_auc = 0
    for key in preds.keys():
        mean_auc += roc_auc(preds[key], targets[key])
    mean_auc /= len(preds.keys())
    return mean_auc


def calculate_auc_with_shuffle(preds, targets):
    assert preds.keys() == targets.keys()

    keys = list(preds.keys())
    num_permutations = 100  # shuffled 100 times

    mean_auc = 0
    for _ in range(num_permutations):
        current_auc = 0
        for key in keys:
            shuffled_pred = preds[key]

            # randomly select a key
            random_target_key = random.choice([k for k in keys if k != key])
            shuffled_target = targets[random_target_key]

            # check the shape and adjust it
            if shuffled_pred.shape != shuffled_target.shape:
                zoom_factor = np.array(shuffled_pred.shape) / np.array(shuffled_target.shape)
                shuffled_target = zoom(shuffled_target, zoom_factor, order=1)

            current_auc += roc_auc(shuffled_pred, shuffled_target)

        mean_auc += current_auc / len(preds.keys())

    mean_auc /= num_permutations
    return mean_auc
