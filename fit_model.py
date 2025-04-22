from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from models.MarketData import MarketData
import pandas as pd

start_date = '2010-01-01'
split_date = '2018-01-01'
end_date = '2020-01-01'
market_data = MarketData("data/tickers")

# Select model
# model = XgboostPredictionModel(market_data.tickers)
model = SingleLstmPredictionModel(market_data.tickers)

features = model.market_to_features_data(market_data)
features.crop_data(start_date, end_date)

train_features, test_features = features.split_by_date(pd.Timestamp(split_date))

model.fit(train_features)
predictions = model.predict(test_features)
metrics = predictions.metrics_df
print(metrics)
metrics.to_csv("lstm_mae.csv")
model.save_model("trained_models/lstm_mae")