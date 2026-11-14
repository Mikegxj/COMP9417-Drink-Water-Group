# COMP9417 · Forecasting NSW Electricity Demand and Price Direction

**UNSW Machine Learning · T3 2026 · Drink Water Group**

At the end of each market day, we forecast electricity demand and price direction in New South Wales (NSW) for the next 48 half-hours. We use **N-BEATSx** for demand forecasting and **relative-price XGBoost** for price classification. We also study how forecast accuracy changes over time and whether retraining helps the models adapt.

## Team Members

| Name | Student ID | Responsibilities | Contact |
| --- | --- | --- | --- |
| Xiaojun Guo | z5698637 | Price direction (classification) | z5698637@ad.unsw.edu.au |
| Jiaqi Huang |  | Price direction (classification) |  |
| Zixin Su |  | Demand(regression) |  |
| Ruiyang Wang |  | Demand(regression) |  |

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

### Demand · N-BEATSx

The PyTorch model uses the previous **10 days of NSW demand** (480 half-hours), Sydney temperature, calendar cycles and NSW public holiday indicators. It predicts all 48 demand values for the next day in one call. We fitted the final demand model using four years of data, from **November 2022 through October 2026**.

### Price Direction · Relative-Price XGBoost

The classifier uses **14 features** derived from time of day, weekday, temperature, historical demand and historical prices. Three features capture how the price at a given half-hour compares with yesterday’s mean. They use the price at that half-hour yesterday, the price at that half-hour last week, and the seven-day average for that half-hour. We scale these differences using yesterday’s price variability.


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
