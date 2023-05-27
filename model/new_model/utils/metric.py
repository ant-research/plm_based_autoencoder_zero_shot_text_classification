import numpy as np
from sklearn import metrics
import logging


def precision_at_k(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores


def compute_scores(probabs, targets, name='test'):
    print('compute scores')
    probabs = np.array(probabs)
    targets = np.array(targets)
   
    preds = np.rint(probabs)  # (probabs >= 0.5)

    accuracy = metrics.accuracy_score(targets, preds)
    f1_score_micro = metrics.f1_score(targets, preds, average='micro')
    f1_score_macro = metrics.f1_score(targets, preds, average='macro')
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} f1 score (micro): {f1_score_micro}")
    print(f"{name} f1 score (macro): {f1_score_macro}")
    precision_at_ks, p5_scores = precision_at_k(targets, probabs)
    print(f"{name} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")
    try:
        auc_score_micro = metrics.roc_auc_score(targets, probabs, average='micro')
        auc_score_macro = metrics.roc_auc_score(targets, probabs, average='macro')
        print(f"{name} auc score (micro): {auc_score_micro}")
        print(f"{name} auc score (macro): {auc_score_macro}")
    except Exception as e:
        pass

    return precision_at_ks[0]

def precision_at_k_single(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = 1
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores


def compute_scores_single(probabs, targets, name='test'):
    print('compute scores single label task')
    probabs = np.array(probabs)
    targets = np.array(targets)
   
    preds = probabs.argmax(1)  # (probabs >= 0.5)
    true_target = targets.argmax(1)
    print(preds[0:10], probabs[0], targets[0:10])
    labels = []
    for i in true_target:
        if i not in labels:
            labels.append(i)
            print(labels)

    accuracy = metrics.accuracy_score(true_target, preds)
    f1_score_micro = metrics.f1_score(true_target, preds, labels=labels, average='micro')
    f1_score_macro = metrics.f1_score(true_target, preds, labels=labels, average='macro')
    weighted_f1_score = metrics.f1_score(true_target, preds, labels=labels, average='weighted')
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} f1 score (micro): {f1_score_micro}")
    print(f"{name} f1 score (macro): {f1_score_macro}")
    print(f"{name} f1 score (weighted): {weighted_f1_score}")
    precision_at_ks, p5_scores = precision_at_k_single(targets, probabs)
    print(f"{name} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")
    print(f"{name} p5 score {p5_scores}\n")
    try:
        auc_score_micro = metrics.roc_auc_score(targets, probabs, average='micro')
        print(f"{name} auc score (micro): {auc_score_micro}")
        auc_score_macro = metrics.roc_auc_score(targets, probabs, average='macro')
        print(f"{name} auc score (macro): {auc_score_macro}")
    except Exception as e:
        pass

    return precision_at_ks[0]
