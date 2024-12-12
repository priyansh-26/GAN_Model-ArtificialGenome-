from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def mia(train_loader: DataLoader, test_loader: DataLoader,
        all_data_size: int, device: torch.device, netD: torch.nn.Module,
        thresholdList: list) -> Tuple[np.ndarray, list]:
    """
    membership inference attack

    Args:
        train_loader (DataLoader): training data
        test_loader (DataLoader): test data
        all_data_size (int): total size of the training data and test data
        device (torch.device): device
        netD (torch.nn.Module): the discriminator of the GAN
        thresholdList (list): list of threshold

    Returns:
        Tuple[np.ndarray, list]: confusion matrix, list of accuracies
    """

    netD.eval()
    predictions = []

    # Predict of training data
    b = 0
    while b < len(train_loader):
        b += 1
        batch_train = next(iter(train_loader))
        real_cpu = batch_train.to(device)
        with torch.no_grad():
            output = netD(real_cpu)
        output = [x for x in output.detach().cpu().numpy()]
        output = list(zip(output, ['train' for _ in range(len(output))]))
        predictions.extend(output)

    # Predict of test data
    b = 0
    while b < len(test_loader):
        b += 1
        batch_test = next(iter(test_loader))
        real_cpu = batch_test.to(device)
        with torch.no_grad():
            output = netD(real_cpu)
        output = [x for x in output.detach().cpu().numpy()]
        output = list(zip(output, ['test' for _ in range(len(output))]))
        predictions.extend(output)

    # Predictions with lower scores come first
    predictions_sorted = sorted(predictions, reverse=True)
    predictions = [x[1]
                   for x in sorted(predictions, reverse=True)]
    print('dataset size: ', len(predictions))

    ## Create a dataframe
    values = [predictions_sorted[i][0].item()
              for i in range(len(predictions_sorted))]
    labels = [predictions_sorted[i][1]
              for i in range(len(predictions_sorted))]
    preds_df = pd.DataFrame((values, labels), index=['value', 'label']).T

    ## Group by and shuffle within each score group
    shuffled = preds_df.groupby('value', group_keys=True).apply(
        lambda x: x.sample(frac=1)).reset_index(drop=True)

    ## Group by sorts in ascending order, so reverse to descending
    shuffled = shuffled.iloc[::-1].reset_index(drop=True)

    # Try different thresholds
    # Take the bottom n% of scores. "Lower scores = predicted as test"
    accuracies = []
    for n in thresholdList:
        train_size = int(all_data_size*n/100)
        accuracy = shuffled.iloc[:train_size]['label']\
            .value_counts(normalize=True)['train']
        # Add to the list
        accuracies.append(accuracy)

    # Calculate false positive rate and true positive rate
    ## Actual labels
    real_label = shuffled['label']
    real_label = real_label.values
    real_label = [1 if x == 'train' else 0 for x in real_label]
    ## Predicted labels
    trains = shuffled['label'].value_counts()['train']
    tests = shuffled['label'].value_counts()['test']
    pred_label = [1] * trains + [0] * tests
    ## Get the confusion matrix
    cm = confusion_matrix(real_label, pred_label)
    tn, fp, fn, tp = cm.flatten()
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    print('tpr:', round(tpr, 6))
    print('fpr:', round(fpr, 6))
    print('Attacker\'s Advantage:', round(abs(tpr-fpr), 6))
    return cm, accuracies
