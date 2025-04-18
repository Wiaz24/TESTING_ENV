from dataloaders import EtfDataloader
from prediction_models import LstmPredictionModel, SingleLstmPredictionModel


from sklearn.model_selection import train_test_split
import pandas as pd

start_date = '2010-02-01'
split_date = '2020-01-03'
end_date = '2025-01-01'

dataloader = EtfDataloader.EtfDataloader("data/tickers")
model = SingleLstmPredictionModel.SingleLstmPredictionModel(dataloader.tickers)
features = model.ohclv_to_features(dataloader.data, start_date, end_date)
model.load_model("trained_models/single_lstm_model.keras")
train_features, test_features = features.split_by_date(pd.Timestamp(split_date))
predictions = model.predict(test_features)
predictions.print_metrics()