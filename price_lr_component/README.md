

```sh
python -m pip install -r requirements.txt
python price_lr_component/train_price_logistic.py data/nem_weather_halfhourly_2010_2026.csv
python check_submission.py price_lr_component
```

To test a zip, put model.py, train.py, price_logistic.py, train_price_logistic.py and artifacts/ at its root, then use:

```sh
python check_submission.py submission.zip
```
