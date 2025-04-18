
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
split_date = '2021-01-04'
end_date = '2025-01-01'
market_data_dir = Path("data/tickers")

market_data = MarketData(market_data_dir)
lstm_model = SingleLstmPredictionModel(market_data.tickers)
lstm_model.load_model("trained_models/single_lstm_model.keras")
features = lstm_model.market_to_features_data(market_data)

market_data.crop_data(split_date, end_date)
features.crop_data(split_date, end_date)

prices = market_data.close_df
X = prices_to_returns(prices)

holding_period = 1  # obliczanie mv co 1 dzień
fitting_period = 60 # obliczanie mv na podstawie ostatnich 60 dni

cv = WalkForward(train_size=fitting_period, test_size=holding_period)

# transaction_cost = 0.001 / holding_period
transaction_cost = 0


# LSTM MV
print("Creating LSTM MV portfolios")
model = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
portfolios: list[MultiPeriodPortfolio] = []
preselected_datas: list[PreselectedCloseData] = []
lstm_predictions = lstm_model.predict(features, verbose=0)

for i in range(5, 11):
    portfolios.append(MultiPeriodPortfolio(name=f"LSTM MV cardinality = {i}"))
    preselected_datas.append(PreselectedCloseData(lstm_predictions, i))


print("Predictions done")

previous_weights = None
for train_idx, test_idx in tqdm(cv.split(X), 
                                desc="Generating LSTM portfolios", 
                                unit="portfolio",
                                total=cv.get_n_splits(X)):
    
    train_data = X.take(train_idx)
    test_data = X.take(test_idx)
    test_date = test_data.index[0]

    for i in range(5, 11):
        preselected_train_data = preselected_datas[i-5].filter_close_df(train_data, test_date)
        preselected_test_data = preselected_datas[i-5].filter_close_df(test_data, test_date)
        model.fit(preselected_train_data)
        portfolio = model.predict(preselected_test_data)
        portfolios[i-5].append(portfolio)

population = Population(portfolios)
fig = population.plot_cumulative_returns()
show(fig)