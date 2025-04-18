
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from models.PreselectedCloseData import PreselectedCloseData
from models.MarketData import MarketData

from plotly.io import show
from pathlib import Path
from skfolio import MultiPeriodPortfolio, Population, Portfolio, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction, InverseVolatility, EqualWeighted, Random
from skfolio.preprocessing import prices_to_returns
import pandas as pd
from tqdm import tqdm

start_date = '2010-01-01'
split_date = '2020-01-04'
end_date = '2025-01-01'
market_data_dir = Path("data/tickers")

market_data = MarketData(market_data_dir)
lstm_model = SingleLstmPredictionModel(market_data.tickers)
lstm_model.load_model("trained_models/single_lstm_model.keras")
features = lstm_model.market_to_features_data(market_data)

xgboost_model = XgboostPredictionModel(market_data.tickers)
xgboost_model.load_model("trained_models")
xgboost_features = xgboost_model.market_to_features_data(market_data)

market_data.crop_data(split_date, end_date)
features.crop_data(split_date, end_date)
xgboost_features.crop_data(split_date, end_date)

prices = market_data.close_df
X = prices_to_returns(prices)

holding_period = 1  # obliczanie mv co 1 dzień
fitting_period = 60 # obliczanie mv na podstawie ostatnich 60 dni

cv = WalkForward(train_size=fitting_period, test_size=holding_period)

# transaction_cost = 0.001 / holding_period
transaction_cost = 0

model1 = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
pred1 = cross_val_predict(model1, X, cv=cv, n_jobs=-1)
pred1.name = "Mean Risk - solver CLARABEL"

model2 = InverseVolatility()
pred2 = cross_val_predict(model2, X, cv=cv, n_jobs=-1)
pred2.name = "Inverse Volatility - no TC"

model3 = EqualWeighted()
pred3 = cross_val_predict(model3, X, cv=cv, n_jobs=-1)
pred3.name = "Equal Weighted - no TC"

model4 = Random()
pred4 = cross_val_predict(model4, X, cv=cv, n_jobs=-1)
pred4.name = "Random - no TC"


# LSTM MV
print("Creating LSTM MV portfolios")
model5 = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
pred5 = MultiPeriodPortfolio(name="LSTM MV")

lstm_predictions = lstm_model.predict(features, verbose=0)
print("Predictions done")
preselected_data = PreselectedCloseData(lstm_predictions, 7)

previous_weights = None
for train_idx, test_idx in tqdm(cv.split(X), 
                                desc="Generating LSTM portfolios", 
                                unit="portfolio",
                                total=cv.get_n_splits(X)):
    
    train_data = X.take(train_idx)
    test_data = X.take(test_idx)
    test_date = test_data.index[0]

    preselected_train_data = preselected_data.filter_close_df(train_data, test_date)
    preselected_test_data = preselected_data.filter_close_df(test_data, test_date)
    model5.fit(preselected_train_data)
    portfolio5 = model5.predict(preselected_test_data)
    pred5.append(portfolio5)

# XGBoost MV
print("Creating Xgboost portfolios")
model6 = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
pred6 = MultiPeriodPortfolio(name="Xgboost MV")

xgboost_predictions = xgboost_model.predict(xgboost_features)
print("Predictions done")
preselected_data = PreselectedCloseData(xgboost_predictions, 7)

previous_weights = None
for train_idx, test_idx in tqdm(cv.split(X), 
                                desc="Generating LSTM portfolios", 
                                unit="portfolio",
                                total=cv.get_n_splits(X)):
    
    train_data = X.take(train_idx)
    test_data = X.take(test_idx)
    test_date = test_data.index[0]

    preselected_train_data = preselected_data.filter_close_df(train_data, test_date)
    preselected_test_data = preselected_data.filter_close_df(test_data, test_date)
    model6.fit(preselected_train_data)
    portfolio6 = model6.predict(preselected_test_data)
    pred6.append(portfolio6)


population = Population([pred1, pred2, pred5, pred6])
fig = population.plot_cumulative_returns()
show(fig)