# Gatorade Color Prediction

This project predicts the next year’s Super Bowl Gatorade color using past years’ data.

It uses:
- Python
- Pandas
- Scikit-learn
- Random Forest classifier

## How it works

- Loads historical Gatorade color data from a CSV file
- Encodes colors into numbers
- Uses the last 3 years of colors as features
- Trains a machine learning model
- Tests the model and prints accuracy
- Predicts the next Gatorade color

## Files

- `super_gatorade_prediction.py` — main script
- `gatorade_colors.csv` — dataset with past Gatorade colors

## How to run

1. Install dependencies:
```bash
pip install pandas scikit-learn
