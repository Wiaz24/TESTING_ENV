from dataloaders import EtfDataloader
from prediction_models import XgboostPredictionModel

from sklearn.model_selection import train_test_split
import pandas as pd

start_date = '2010-01-01'
split_date = '2020-01-03'
end_date = '2025-01-01'

dataloader = EtfDataloader.EtfDataloader("data/tickers", start_date, end_date)
model = XgboostPredictionModel.XgboostPredictionModel(dataloader.tickers)
features = model.ohclv_to_features(dataloader.data)

train_data, test_data = dataloader.split_train_test_by_date(features, split_date)

X_train, y_train = dataloader.get_X_y(train_data, target_column='next_day_log_return')
model.fit(X_train, y_train)

X_test, y_test = dataloader.get_X_y(test_data, target_column='next_day_log_return')
predictions = model.evaluate(X_test, y_test)
