# Member 3 - ML Models (Fake Reviews)

Naive Bayes, Logistic Regression and SVM (LinearSVC) on TF-IDF (1-2 grams).

## Run
```bash
pip install -r requirements.txt
python train_models.py --data data/cleaned_reviews.csv          # fast, default params
python train_models.py --data data/cleaned_reviews.csv --tune   # 3-fold grid search
```
Use the Phase 1 cleaned CSV (text column auto-detected: clean_text / text_clean / text_ / text).
Label column defaults to `label` (CG = fake -> 1, OR = real -> 0). Override with `--text-col` / `--label-col`.

## Outputs (`outputs/`)
- `models/*.joblib` - trained pipelines (TF-IDF + classifier), load and call `.predict([text])`
- `model_comparison.csv` / `.png` - accuracy, precision, recall, F1, ROC-AUC, timings
- `confusion_matrices.png`

## Notes
- 80/20 stratified split, `random_state=42`.
- TF-IDF lives inside each Pipeline, so it is fit on train only (no leakage).
