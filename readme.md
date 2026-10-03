# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## Project Structure

- `data/`: training, validation, template, and December input CSV files
- `src/train_and_predict.py`: production training and prediction pipeline
- `src/model_comparison.py`: time-based model bake-off
- `score.py`: company evaluation script
- `notebooks/`: space for exploratory analysis

## Setup

Create or activate the project virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train and Predict

Run the production pipeline from the project root:

```bash
python src/train_and_predict.py
```

The pipeline cleans negative weights, imputes missing numeric values, creates calendar features, and uses the time-based split below:

- Training: dates before `2025-10-01`
- Internal validation: dates from `2025-10-01` onward

The final model is `HistGradientBoostingRegressor`, selected after comparing it with Ridge, Random Forest, and XGBoost. Categorical features are ordinal-encoded with unknown values mapped to `-1`.

The command creates or updates:

- `validation_predictions.csv`
- `data/december-chart-inputs.csv`

## Model Comparison

Run the empirical model comparison with the same cleaning and time-based holdout:

```bash
python src/model_comparison.py
```

The script prints an ASCII table containing each model's MAE, RMSE, and runtime. The latest comparison selected HistGradientBoosting based on the lowest holdout MAE.

## Score Outputs

Run the company scorer after generating predictions:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
