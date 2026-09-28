# Kepler Exoplanet Classification

Machine learning pipeline that classifies objects from NASA's Kepler mission as **confirmed exoplanets** or **false positives**, using the Kepler Objects of Interest (KOI) catalog.

The project is based on the paper *Using Machine Learning for Exoplanet Classification* (Research Archive of Rising Scholars), which tuned a single MLP neural network. Here, that approach is compared against stronger tree-based models on the same dataset.

## Dataset

[NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) on Kaggle (`cumulative.csv`).

- Only `CONFIRMED` (label 1) and `FALSE POSITIVE` (label 0) rows are used. `CANDIDATE` rows are excluded.
- Label-like columns (`koi_score`, `koi_fpflag_*`, IDs) are dropped to avoid target leakage.
- Columns with more than 40% missing values are dropped, and the remaining gaps are filled with training-set medians.

The dataset is not included in this repo. Download it from Kaggle.

## Models

| Model | Why it is included |
|---|---|
| Logistic Regression | Simple baseline: can a straight-line boundary separate planets from noise? |
| Random Forest | Many decision trees voting together; captures curved, non-linear patterns and resists noise. |
| XGBoost | Boosted trees: each new tree corrects the mistakes of the previous ones. |
| LightGBM | Also boosted trees, grown leaf-wise for speed. |

Random Forest, XGBoost, and LightGBM are tuned with `RandomizedSearchCV` (3-fold CV, F1 scoring). The best model by test F1 is saved as a scikit-learn pipeline (scaler + classifier).

## Metrics

Accuracy, precision, recall, F1-score, and ROC-AUC on a stratified 80/20 hold-out split (`random_state=42`).

## Results

Current pipeline, single 80/20 split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 91.05% | 87.10% | 83.88% | 85.46% | 96.70% |
| Random Forest | 93.92% | 91.48% | 88.89% | 90.17% | 97.97% |
| XGBoost | 94.67% | 93.20% | 89.54% | 91.33% | 98.59% |
| LightGBM | 94.67% | 93.39% | 89.32% | 91.31% | 98.57% |

XGBoost and LightGBM are effectively tied; the gap between them is within noise on a test set of this size.

### Known limitations

- Results come from a single train/test split, so there are no confidence intervals yet.
- The final models are trained on a 100,000-row bootstrap resample of the ~5.8k real training rows, which adds no new information.
- The saved pipeline has no imputer, so raw rows with missing values are not handled the way they were in training.

Planned improvements: repeated stratified cross-validation, training on real rows only with class weights, an imputer inside the pipeline, feature ablation (uncertainty and positional columns), a paper-style MLP baseline on the same split, SHAP explanations, and ranking of the `CANDIDATE` objects.

## Project structure

```
.
├── Kepler_ExoPlanet.ipynb   # Colab notebook
├── kepler_exoplanet.py      # same code exported as a script
├── requirements.txt
└── README.md
```

## How to run

### In Google Colab (as written)

1. Download `cumulative.csv` from Kaggle and upload it to your Google Drive (`MyDrive/`).
2. Open `Kepler_ExoPlanet.ipynb` in Colab.
3. Update `file_path` if your CSV is somewhere else, then run all cells.

### Locally

```bash
pip install -r requirements.txt
```

The code uses `google.colab` to mount Drive and download the model. To run locally, remove those lines (`drive.mount`, `files.download`) and set `file_path` to the local path of `cumulative.csv`.

## Contributors

- Aliza Zaman

## Acknowledgements

- NASA Kepler mission and the NASA Exoplanet Archive for the data.
- Kaggle dataset: `nasa/kepler-exoplanet-search-results`.
