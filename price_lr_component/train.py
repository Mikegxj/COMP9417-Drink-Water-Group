"""INTEGRATION EXAMPLE: final train.py must also retrain the demand model.
Usage: python train.py path/to/nem_weather_halfhourly_2010_2026.csv
"""
import os
import subprocess
import sys

if __name__ == '__main__':
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'train_price_logistic.py'),
                    sys.argv[1]], check=True)
