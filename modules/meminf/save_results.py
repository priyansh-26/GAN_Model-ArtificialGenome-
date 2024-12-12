import os

import matplotlib.pyplot as plt
import numpy as np


def draw_accuracy_bar(
    accuracies: list, thresholdList: list, test_size: int, all_size: int,
    model_type: str, out_dir: str, is_white: bool):
    """
    Represent the accuracy of membership inference with a bar chart

    Args:
        accuracies (list): List of accuracies
        thresholdList (list): List of thresholds
        test_size (int): Size of the test data
        all_size (int): Size of all data (train + test)
        model_type (str): Type of model
        out_dir (str): Output directory
        is_white (bool): Whether the attack type is white-box
    """
    # Prepare the graph
    fig = plt.figure(figsize=(5, 6))

    # Create the bar chart
    left = [x+1 for x in range(len(accuracies))]
    plt.bar(left, accuracies, align='center', alpha=0.7, color='grey',
            width=0.5, tick_label=[str(x)+'%' for x in thresholdList])
    for i in range(len(accuracies)):
        plt.annotate(round(accuracies[i], 2), (i+1, accuracies[i]))
    plt.xlabel('Percentage')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, 0.2))

    # Draw baseline in red
    baseline = (all_size - test_size) / all_size
    plt.axhline(y=baseline, color='red', linestyle='--')

    # Graph title and file name based on the attack type
    if is_white:
        plt.title(f'White-box attack ({model_type})')
        fig.savefig(os.path.join(out_dir, f'wb_accuracy_{model_type}.pdf'), format='pdf')
    else:
        plt.title(f'Black-box attack ({model_type})')
        fig.savefig(os.path.join(out_dir, f'bb_accuracy_{model_type}.pdf'), format='pdf')
    plt.close()


def save_mia_results_as_txt(
    cm: np.ndarray, accuracies: list, thresh_list: list, filepath: str):
    """
    Calculate some metrics from the confusion matrix and save them as a text file

    Args:
        cm (np.ndarray): confusion matrix
        accuracies (list): list of accuracies
        thresh_list (list): list of threshold
        filepath (str): text file path
    """
    tn, fp, fn, tp = cm.flatten()
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    with open(filepath, 'w') as f:
        f.write(f'True Negative: {tn}\n')
        f.write(f'False Positive: {fp}\n')
        f.write(f'False Negative: {fn}\n')
        f.write(f'True Positive: {tp}\n')
        f.write('------------------------------------\n')
        f.write(f'FPR: {fpr}\n')
        f.write(f'TPR: {tpr}\n')
        f.write(f"Attacker's Advantage: {abs(tpr-fpr)}\n")
        f.write('------------------------------------\n')
        for i, thresh in enumerate(thresh_list):
            f.write(f'Top {thresh}% Accuracy: {accuracies[i]}\n')
    f.close()