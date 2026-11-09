# Results

Windows: 4643 from 40 recordings.
- Normal: 409 windows from 4 recordings
- Inner Race: 1410 windows from 12 recordings
- Outer Race: 1413 windows from 12 recordings
- Ball: 1411 windows from 12 recordings

## Leave-one-load-out

| Model | Accuracy (pooled) | Macro F1 (pooled) | Worst-fold accuracy | Worst-fold macro F1 |
|---|---|---|---|---|
| Logistic regression, 4 hand-made features | 72.5% | 0.767 | 61.7% | 0.684 |
| Logistic regression, FFT spectrum | 98.5% | 0.987 | 95.5% | 0.964 |
| Neural network (PyTorch MLP), FFT spectrum | 98.5% | 0.988 | 96.3% | 0.971 |

Per fold (accuracy / macro F1):

- Logistic regression, 4 hand-made features: test on 0 HP: 61.7% / 0.684, test on 1 HP: 81.3% / 0.843, test on 2 HP: 73.9% / 0.779, test on 3 HP: 72.4% / 0.753
- Logistic regression, FFT spectrum: test on 0 HP: 95.5% / 0.964, test on 1 HP: 99.5% / 0.996, test on 2 HP: 99.7% / 0.997, test on 3 HP: 99.3% / 0.992
- Neural network (PyTorch MLP), FFT spectrum: test on 0 HP: 96.3% / 0.971, test on 1 HP: 99.2% / 0.994, test on 2 HP: 98.7% / 0.989, test on 3 HP: 99.7% / 0.998

Confusion matrices (rows = true class, columns = predicted; order Normal, Inner Race, Outer Race, Ball):

- Logistic regression, 4 hand-made features: `[[401, 0, 8, 0], [0, 851, 411, 148], [0, 105, 896, 412], [11, 122, 62, 1216]]`
- Logistic regression, FFT spectrum: `[[409, 0, 0, 0], [0, 1345, 16, 49], [3, 0, 1410, 0], [0, 0, 0, 1411]]`
- Neural network (PyTorch MLP), FFT spectrum: `[[409, 0, 0, 0], [0, 1343, 30, 37], [0, 0, 1413, 0], [0, 0, 1, 1410]]`

## Leave-one-fault-size-and-load-out (stricter)

| Model | Accuracy (pooled) | Macro F1 (pooled) | Worst-fold accuracy | Worst-fold macro F1 |
|---|---|---|---|---|
| Logistic regression, 4 hand-made features | 51.8% | 0.508 | 16.6% | 0.128 |
| Logistic regression, FFT spectrum | 57.6% | 0.524 | 42.0% | 0.408 |
| Neural network (PyTorch MLP), FFT spectrum | 56.7% | 0.521 | 42.0% | 0.410 |

Per fold (accuracy / macro F1):

- Logistic regression, 4 hand-made features: test on 0.007 in at 0 HP: 64.5% / 0.654, test on 0.007 in at 1 HP: 74.9% / 0.721, test on 0.007 in at 2 HP: 64.6% / 0.632, test on 0.007 in at 3 HP: 65.8% / 0.643, test on 0.014 in at 0 HP: 34.7% / 0.271, test on 0.014 in at 1 HP: 56.2% / 0.502, test on 0.014 in at 2 HP: 62.3% / 0.553, test on 0.014 in at 3 HP: 47.0% / 0.339, test on 0.021 in at 0 HP: 44.7% / 0.522, test on 0.021 in at 1 HP: 47.5% / 0.463, test on 0.021 in at 2 HP: 41.7% / 0.376, test on 0.021 in at 3 HP: 16.6% / 0.128
- Logistic regression, FFT spectrum: test on 0.007 in at 0 HP: 65.5% / 0.642, test on 0.007 in at 1 HP: 72.3% / 0.655, test on 0.007 in at 2 HP: 73.6% / 0.660, test on 0.007 in at 3 HP: 66.9% / 0.627, test on 0.014 in at 0 HP: 55.7% / 0.505, test on 0.014 in at 1 HP: 50.0% / 0.415, test on 0.014 in at 2 HP: 51.0% / 0.425, test on 0.014 in at 3 HP: 64.0% / 0.589, test on 0.021 in at 0 HP: 42.0% / 0.408, test on 0.021 in at 1 HP: 49.3% / 0.414, test on 0.021 in at 2 HP: 49.8% / 0.416, test on 0.021 in at 3 HP: 49.9% / 0.417
- Neural network (PyTorch MLP), FFT spectrum: test on 0.007 in at 0 HP: 48.2% / 0.543, test on 0.007 in at 1 HP: 74.5% / 0.665, test on 0.007 in at 2 HP: 74.8% / 0.666, test on 0.007 in at 3 HP: 55.4% / 0.551, test on 0.014 in at 0 HP: 65.8% / 0.595, test on 0.014 in at 1 HP: 50.0% / 0.416, test on 0.014 in at 2 HP: 53.5% / 0.475, test on 0.014 in at 3 HP: 66.0% / 0.608, test on 0.021 in at 0 HP: 42.0% / 0.410, test on 0.021 in at 1 HP: 48.8% / 0.412, test on 0.021 in at 2 HP: 49.8% / 0.416, test on 0.021 in at 3 HP: 49.9% / 0.415

Confusion matrices (rows = true class, columns = predicted; order Normal, Inner Race, Outer Race, Ball):

- Logistic regression, 4 hand-made features: `[[1038, 45, 86, 58], [0, 408, 715, 287], [194, 549, 394, 276], [294, 62, 65, 990]]`
- Logistic regression, FFT spectrum: `[[1227, 0, 0, 0], [67, 126, 734, 483], [470, 552, 391, 0], [4, 2, 4, 1401]]`
- Neural network (PyTorch MLP), FFT spectrum: `[[1227, 0, 0, 0], [3, 187, 745, 475], [470, 658, 282, 3], [2, 3, 6, 1400]]`
