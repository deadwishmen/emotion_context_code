import numpy as np
from sklearn.metrics import average_precision_score

def test_scikit_ap(preds, labels):
    ap = np.zeros((preds.shape[0]))
    for i in range(preds.shape[0]):
        ap[i] = average_precision_score(labels[i, :], preds[i, :])
    print('ap', ap, ap.shape, ap.mean())
    return ap.mean()

def test_emotic_vad(preds, labels):
    mae = np.mean(np.abs(preds - labels), axis=1)
    print('MAE per VAD:', mae)
    return mae.mean()