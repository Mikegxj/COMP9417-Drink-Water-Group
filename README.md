# COMP9417 · Forecasting NSW Electricity Demand and Price Direction

**UNSW Machine Learning · T3 2026 · Drink Water Group**

At the end of each market day, we forecast electricity demand and price direction in New South Wales (NSW) for the next 48 half-hours. We use an **MLP ensemble** for demand forecasting and **relative-price XGBoost** for price classification, and evaluate how retraining affects forecasting performance over time. We also study how forecast accuracy changes over time and whether retraining helps the models adapt.

## Team Members

| Name | Student ID | Responsibilities | Contact |
| --- | --- | --- | --- |
| Xiaojun Guo | z5698637 | Price direction (classification) | z5698637@ad.unsw.edu.au |
| Jiaqi Huang | z5773204 | Price direction (classification) | z5773204@ad.unsw.edu.au |
| Zixin Su | z5535967 | Demand (regression) | z5535967@ad.unsw.edu.au |
| Ruiyang Wang | z5638600 | Demand (regression) | z5638600@ad.unsw.edu.au |

## Project Structure

| Directory | Contents |
| --- | --- |
| [`Final_submission/`](Final_submission/) | Final submission materials |
| [`data/`](data/) | Project data and dataset information |
| [`docs/demand/`](docs/demand/) | Documentation for the demand forecasting experiments |
| [`example_package/`](example_package/) | Course example showing how to package a forecaster |
| [`experiments/`](experiments/) | Forecasting experiments and model comparisons |
| [`harness/`](harness/) | Course utilities for preparing forecasts, running models and scoring predictions |
| [`results/`](results/) | Evaluation tables, figures and experiment outputs |
| [`specs/`](specs/) | Course project specifications |
| [`src/`](src/) | Project source code and shared feature utilities |
| [`submission/`](submission/) | Runnable forecasting package with `model.py`, `train.py` and fitted artifacts |
| [`template/`](template/) | Course template for the required forecasting interface |

## Dataset and Prediction Tasks

The course dataset contains half-hourly electricity demand and spot prices for NSW, Victoria and Queensland, together with weather data for Sydney, Melbourne and Brisbane. It covers **January 2010 through October 2026**. Timestamps mark the end of each half-hour in Australian Eastern Standard Time (AEST).

For each forecast, the package receives market history through day D and weather for day D+1. It returns 48 rows with two predictions per row.

| Task | Output | Forecast target | Metrics |
| --- | --- | --- | --- |
| Demand forecasting | `nsw_demand_mw` | NSW electricity demand in MW for each half-hour of D+1 | MAE and MAPE |
| Price classification | `nsw_price_up` | `1` if the half-hourly NSW spot price exceeds the **mean NSW price on day D**, otherwise `0` | Balanced accuracy and accuracy |

## Forecasting Models

### Demand · MLP Ensemble

The demand forecaster averages predictions from **five multilayer perceptrons (MLPs)** trained with random seeds 0–4. Each network predicts all 48 half-hourly demand values for the next day.

The model uses **402 input values**, including demand at the same half-hours yesterday and last week, Sydney weather, weekday indicators, NSW public holidays and seasonal features. Historical demand inputs and prediction targets are normalised by the mean demand over the previous seven days. Predictions are converted back to MW using that same mean.

Each network has the architecture **402 → 256 → 256 → 48**, with GELU activations and dropout of 0.1. Input standardisation is fitted using the training window only.

| Setting | Value |
| --- | --- |
| Ensemble | Five MLPs, seeds 0–4 |
| Hidden layers | Two layers of 256 units |
| Activation | GELU |
| Dropout | 0.1 |
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Training epochs | 55 |
| Batch size | 64 |
| Training loss | Mean absolute percentage error |
| Final training window | Most recent four years of complete market days |

The final training procedure follows the periodic-retraining strategy selected during historical evaluation. If demand prediction raises an exception or produces invalid values, the package uses demand from the same half-hours yesterday.

### Price Direction · Relative-Price XGBoost

Our price forecaster uses an **XGBoost binary classifier** to predict whether the NSW spot price in each half-hour of day D+1 will exceed the **mean NSW price on day D**. The model produces 48 predictions, with `1` indicating a price above that reference and `0` indicating a price at or below it.

#### Input Features

The classifier uses **14 features** combining calendar information, forecast-day temperatures and historical market observations.

| Feature group | Features | Count |
| --- | --- | ---: |
| Calendar | Half-hour slot, weekday sine and weekday cosine | 3 |
| Temperature | Sydney, Melbourne and Brisbane temperatures for the target half-hour | 3 |
| Historical demand | NSW demand at the same half-hour yesterday and seven days earlier | 2 |
| Historical price indicators | Whether the NSW price at the same half-hour yesterday and seven days earlier exceeds yesterday’s mean price | 2 |
| Price variability | Standard deviation of NSW prices over yesterday’s 48 half-hours | 1 |
| Relative prices | Yesterday’s price, last week’s price and the seven-day average at the same half-hour, each expressed relative to yesterday’s mean | 3 |
| **Total** | | **14** |

For each relative-price feature, we subtract yesterday’s mean NSW price from the historical price value and divide by yesterday’s price standard deviation, with a minimum denominator of 1.0. The seven-day average uses the same half-hour across the seven preceding days.

This representation describes historical prices relative to the recent market level. It was designed to reduce sensitivity to changes in absolute price levels over time. All market-derived features use observations available by the end of day D, while target-day temperatures come from the supplied weather inputs.

#### Model Configuration

| Setting | Value |
| --- | --- |
| Model | XGBoost classifier |
| Objective | Binary logistic |
| Number of trees | 300 |
| Maximum tree depth | 3 |
| Minimum child weight | 5 |
| Learning rate | 0.05 |
| Tree construction | Histogram-based |
| Classification threshold | 0.5 |
| Random seed | 42 |


## Run the Project

Run the following commands from the course kit root using **Python 3.11**. Place the released dataset at `data/nem_weather_halfhourly_2010_2026.csv`.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the fitted models

With the fitted artifacts in `submission/artifacts/`, run the course checker on the combined package.

```bash
python check_submission.py submission --data data/nem_weather_halfhourly_2010_2026.csv
```

By default, the checker runs both models on the final **14 complete days** and compares their predictions with the two baselines. These dates fall within the final training windows, so the scores are in-sample. Use this check to verify the package and the historical results above to assess price forecasting performance over time.

### 3. Rebuild the fitted models

To reproduce the demand and price artifacts from the released CSV, run the training script and then repeat the package check.

```bash
python submission/train.py data/nem_weather_halfhourly_2010_2026.csv
```
