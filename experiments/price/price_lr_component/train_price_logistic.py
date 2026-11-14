"""Train the price-direction logistic regression using chronological validation.

Usage: python train_price_logistic.py path/to/nem_weather_halfhourly_2010_2026.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from price_logistic import make_historical_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'artifacts', 'price_lr.joblib'))
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path, parse_dates=['timestamp'], index_col='timestamp').sort_index()
    X, y, dates = make_historical_table(df, '2010-01-01', '2014-12-31')
    train = dates.year <= 2013
    val = dates.year == 2014

    def build_model():
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42))

    model = build_model()
    model.fit(X.loc[train], y[train])
    predicted = model.predict(X.loc[val])
    print('2014 validation samples:', int(val.sum()))
    print('2014 actual positive proportion:', round(float(np.mean(y[val])), 4))
    print('2014 predicted positive proportion:', round(float(np.mean(predicted)), 4))
    print('2014 LR accuracy (%):', round(accuracy_score(y[val], predicted) * 100, 3))
    print('2014 LR balanced accuracy (%):', round(balanced_accuracy_score(y[val], predicted) * 100, 3))

    # After choosing the method using 2014, refit on 2010-2014 (Stage 1 frozen model).
    final_model = build_model()
    final_model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    joblib.dump(final_model, args.output)
    print('Saved 2010-2014 frozen model:', args.output)
    print('Important: Stage 2/3 adaptation and final refitting are NOT implemented here.')


if __name__ == '__main__':
    main()
