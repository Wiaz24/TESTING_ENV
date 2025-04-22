
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from models.PreselectedTickers import PreselectedTickers
from models.MarketData import MarketData
from models.WorstCaseOmega import WorstCaseOmega
from models.PredictionBasedWorstCaseOmega import PredictionBasedWorstCaseOmega

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
lstm_model.load_model("trained_models/single_lstm/single_lstm_model.keras")
features = lstm_model.market_to_features_data(market_data)

xgboost_model = XgboostPredictionModel(market_data.tickers)
xgboost_model.load_model("trained_models/xgboost_normalized")
xgboost_features = xgboost_model.market_to_features_data(market_data)

market_data.crop_data(split_date, end_date)
# features.crop_data(split_date, end_date)
xgboost_features.crop_data(split_date, end_date)

prices = market_data.close_df
X = prices_to_returns(prices)

holding_period = 1  # obliczanie mv co 1 dzień
fitting_period = 60 # obliczanie mv na podstawie ostatnich 60 dni

cv = WalkForward(train_size=fitting_period, test_size=holding_period)

transaction_cost = 0.001 / holding_period
# transaction_cost = 0

models: list[ObjectiveFunction] = [
    MeanRisk(risk_measure=RiskMeasure.VARIANCE),
    WorstCaseOmega(delta=0.8),
    # InverseVolatility(),
    EqualWeighted()
    # Random()
]
mpportfolios: Population = Population([])

# print("Creating all asset portfolios")
# for model in models:
#     portfolio = cross_val_predict(model, X, cv=cv, n_jobs=-1)
#     portfolio.name = f"All asstes - {model.__class__.__name__}"
#     mpportfolios.append(portfolio)

# LSTM MV
print("Creating LSTM preselected portfolios")
lstm_predictions = lstm_model.predict(features, verbose=0)
preselected_tickers = PreselectedTickers(lstm_predictions)

for model in models:
    mpportfolio = MultiPeriodPortfolio(name=f"LSTM preselected - {model.__class__.__name__}")

    for train_idx, test_idx in tqdm(cv.split(X), 
                                    desc="Generating portfolios", 
                                    unit="portfolio",
                                    total=cv.get_n_splits(X)):
        train_data = X.take(train_idx)
        test_data = X.take(test_idx)
        test_date = test_data.index[0]

        preselected_train_data = preselected_tickers.filter_close_df(train_data, test_date)
        preselected_test_data = preselected_tickers.filter_close_df(test_data, test_date)

        model.fit(preselected_train_data)
        portfolio = model.predict(preselected_test_data)
        mpportfolio.append(portfolio)
    mpportfolios.append(mpportfolio)

model = PredictionBasedWorstCaseOmega()
mpportfolio = MultiPeriodPortfolio(name=f"LSTM preselected - {model.__class__.__name__}")
for train_idx, test_idx in tqdm(cv.split(X), 
                                    desc="Generating portfolios", 
                                    unit="portfolio",
                                    total=cv.get_n_splits(X)):
    train_data = X.take(train_idx)
    test_data = X.take(test_idx)
    test_date = test_data.index[0]

    preselected_train_data = preselected_tickers.filter_close_df(train_data, test_date)
    preselected_test_data = preselected_tickers.filter_close_df(test_data, test_date)
   
    preselected_predictions = lstm_predictions.predictions_df.loc[train_data.index]
    preselected_predictions = preselected_tickers.filter_close_df(preselected_predictions, test_date)
    model.fit(preselected_train_data, predictions=preselected_predictions)
    portfolio = model.predict(preselected_test_data)
    mpportfolio.append(portfolio)
mpportfolios.append(mpportfolio)


# XGBoost MV
# print("Creating Xgboost preselected portfolios")
# xgboost_predictions = xgboost_model.predict(xgboost_features)
# preselected_tickers = PreselectedTickers(xgboost_predictions)
# for train_idx, test_idx in tqdm(cv.split(X), 
#                                 desc=f"Generating portfolios", 
#                                 unit="portfolio",
#                                 total=cv.get_n_splits(X)):
    
#     train_data = X.take(train_idx)
#     test_data = X.take(test_idx)
#     test_date = test_data.index[0]

#     preselected_train_data = preselected_tickers.filter_close_df(train_data, test_date)
#     preselected_test_data = preselected_tickers.filter_close_df(test_data, test_date)

#     for model in models:
#         portfolio = MultiPeriodPortfolio(name=f"Xgboost preselected - {model.__class__.__name__}")
#         model.fit(preselected_train_data)
#         portfolio = model.predict(preselected_test_data)
#         mpportfolios.append(portfolio)

#     model = PredictionBasedWorstCaseOmega()
#     preselected_predictions = xgboost_predictions.predictions_df.loc[train_data.index]
#     preselected_predictions = preselected_tickers.filter_close_df(preselected_predictions, test_date)
#     model.fit(preselected_train_data, predictions=preselected_predictions)
#     portfolio = model.predict(preselected_test_data)
#     portfolio.name = f"Xgboost preselected - {model.__class__.__name__}"
#     mpportfolios.append(portfolio)


fig = mpportfolios.plot_cumulative_returns()
show(fig)