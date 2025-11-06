# Evaluation Results Directory

This directory contains the comprehensive evaluation results of the Diabetic Retinopathy detection model.

## Generated Files

After running `python ml/evaluate.py`, this directory will contain:

### Metrics & Reports
- **`metrics.json`** - Comprehensive evaluation metrics in JSON format
  - Overall accuracy, Cohen's Kappa
  - Per-class precision, recall, F1-score, AUC
  - Sensitivity and specificity for each class
  - Confusion matrix

- **`classification_report.txt`** - Detailed classification report
  - Formatted text report with all per-class metrics
  - Overall accuracy and kappa scores
  - Suitable for inclusion in academic reports

- **`summary.txt`** - Quick summary of evaluation results
  - High-level overview
  - Key performance indicators
  - Easy-to-read format

### Predictions
- **`test_predictions.csv`** - All test set predictions
  - Image IDs
  - True labels and predicted labels
  - Confidence scores
  - Class probabilities for all 5 DR severity levels
  - Correctness indicator

### Visualizations
- **`confusion_matrix.png`** - Visual confusion matrix
  - Shows classification patterns
  - Highlights misclassifications
  - Color-coded for easy interpretation
  - 300 DPI, publication-quality

- **`roc_curves.png`** - ROC curves for all classes
  - One curve per DR severity class
  - AUC scores displayed
  - Helps assess classifier performance
  - 300 DPI, publication-quality

## How to Generate Results

```bash
# From project root directory
python ml/evaluate.py --labels-csv ml/data/train.csv \
                      --img-dir ml/data/train_images \
                      --model-path ml/models/best_model-1.pth \
                      --output-dir results
```

## How to Use Results

### For Your Final Year Report

1. **Metrics Table:**
   - Copy data from `summary.txt` or `classification_report.txt`
   - Create formatted tables for your report

2. **Visualizations:**
   - Include `confusion_matrix.png` and `roc_curves.png` in your report
   - Files are 300 DPI and publication-ready

3. **Discussion:**
   - Use `EVALUATION_RESULTS.md` template to structure your analysis
   - Fill in interpretations and clinical relevance

### For Presentations

- Use visualizations from PNG files
- Extract key metrics from `summary.txt`
- Highlight overall accuracy and per-class performance

### For Further Analysis

- Load `test_predictions.csv` in Excel/Python for additional analysis
- Identify specific failure cases
- Analyze prediction confidence distribution
- Compare performance across different DR severity levels

## Understanding the Metrics

### Overall Metrics

- **Accuracy:** Percentage of correct predictions (all classes)
- **Cohen's Kappa:** Agreement measure accounting for chance
  - 0.81-1.00: Almost perfect agreement
  - 0.61-0.80: Substantial agreement
  - 0.41-0.60: Moderate agreement
  - 0.21-0.40: Fair agreement
  - < 0.20: Poor agreement

### Per-Class Metrics

- **Precision:** Of all predictions for a class, how many were correct?
- **Recall (Sensitivity):** Of all actual cases, how many did we detect?
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under ROC curve (0.5 = random, 1.0 = perfect)
- **Specificity:** True negative rate

### Medical Relevance

For Diabetic Retinopathy screening:
- **High Sensitivity (Recall) is critical** for severe cases - don't miss serious conditions
- **High Specificity is important** for mild cases - reduce false alarms
- **AUC close to 1.0** indicates good discrimination ability

## Troubleshooting

### No files in this directory?
- Run the evaluation script first: `python ml/evaluate.py`
- Check that the model path is correct
- Ensure dataset paths are valid

### Files look different than expected?
- Results depend on your specific model and dataset
- Different random seeds may produce slightly different splits
- Class imbalance will affect per-class metrics

### Want to regenerate results?
- Simply run the evaluation script again
- Previous files will be overwritten
- Use `--output-dir` to create separate result folders

## Next Steps

1. ✅ Run evaluation script
2. ✅ Review generated files
3. ✅ Fill in `EVALUATION_RESULTS.md` template
4. ✅ Include results in final year report
5. ✅ Prepare discussion of findings
6. ✅ Address any performance issues identified

---

**Note:** This directory is created automatically when you run the evaluation script. If it doesn't exist yet, the script will create it.
