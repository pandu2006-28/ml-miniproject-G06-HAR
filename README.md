# Classifying Human Activity Using Sensor Data (PAMAP2)

Mini-project: classify 12 physical activities from wearable inertial-sensor data (PAMAP2) and compare four
machine-learning models with a **subject-wise** train/test split (no subject appears in both).

## Files
| File | Purpose |
|---|---|
| `pamap2_har_complete.py` | Full pipeline: download, windowing, features, 4 models, evaluation, outputs |
| `demo_predict.py` | Live demo: predicts activities for real windows of a held-out subject |
| `pamap2_har_colab.ipynb` | Ready-to-run Colab notebook |
| `requirements.txt` | Python dependencies |

## Setup and run (Google Colab, recommended)
1. Open `pamap2_har_colab.ipynb` in Colab (File -> Upload notebook), click **Connect**.
2. Run the cells in order. The script downloads PAMAP2 automatically (~690 MB); no manual upload is needed.
   Takes about 3 minutes on a free CPU runtime.
3. Results are written to `/content/pamap2_har/outputs/`.

## Setup and run (local machine)
```bash
pip install -r requirements.txt
python pamap2_har_complete.py          # downloads the dataset to ./pamap2_har and trains
python demo_predict.py                 # live demo (after training)
```
Optional: `PAMAP2_CV_FOLDS=2 python pamap2_har_complete.py` for faster cross-validation.
If the download fails, place `PAMAP2_Dataset.zip` at `pamap2_har/pamap2.zip` and rerun.

## Method
- Data: 3 IMUs (hand, chest, ankle) at 100 Hz; 39 sensor channels (heart rate and orientation excluded).
- Windows: 2 s (200 samples), 50 % overlap, never crossing an activity change or timestamp gap.
- Features: mean, std, min, max, median, IQR, RMS per channel = 273 features.
- Split: train subjects 101-106, test subjects 107-108 (109 excluded).
- Models: Random Forest, Extra Trees, Logistic Regression, Linear SVM. The best model is chosen by
  subject-grouped cross-validation on the training subjects only, never on the test set.

## Outputs
`model_comparison.csv`, `classification_report.txt`, `confusion_matrix.png`, `feature_importance.csv`,
`top_features.png`, `best_pamap2_model.joblib`, `test_predictions.csv`, `run_summary.txt`

A copy of the main results from our Colab run is in `results/`.

## Results (test subjects 107-108)
| Model | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Extra Trees** | 0.778 | 0.934 | 0.927 |
| Random Forest | 0.733 | 0.872 | 0.869 |
| Logistic Regression | 0.667 | 0.715 | 0.677 |
| Linear SVM | 0.578 | 0.685 | 0.663 |

## Team
Sec-G, Team G-06 (Team No. 06), Problem No. 2

- PANDU C V (PES1UG24AM382)
- SUMAN DIDDIMANI (PES1UG24AM409)
