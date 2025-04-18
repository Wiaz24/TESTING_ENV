from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from models.MarketData import MarketData

from sklearn.model_selection import train_test_split
import pandas as pd

start_date = '2010-02-01'
split_date = '2020-01-03'
end_date = '2025-01-01'
market_data = MarketData("data/tickers")

# Select model
model = XgboostPredictionModel(market_data.tickers)

features = model.market_to_features_data(market_data)
features.crop_data(start_date, end_date)

train_features, test_features = features.split_by_date(pd.Timestamp(split_date))

model.fit(train_features)
predictions = model.predict(test_features)
print(predictions.metrics_df)
model.save_model("trained_models")