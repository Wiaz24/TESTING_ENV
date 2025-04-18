
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
grid_model = XgboostPredictionModel(market_data.tickers)
grid_model.load_model("trained_models")

ifa_model = XgboostPredictionModel(market_data.tickers)
ifa_model.load_model("trained_models/xgboost_ifa")
features = grid_model.market_to_features_data(market_data)

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
print("Creating Xgboost portfolios")
model = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
portfolios: list[MultiPeriodPortfolio] = []
preselected_datas: list[PreselectedCloseData] = []
grid_predictions = grid_model.predict(features)
ifa_predictions = ifa_model.predict(features)


portfolios.append(MultiPeriodPortfolio(name=f"Xgboost grid search MV"))
preselected_datas.append(PreselectedCloseData(grid_predictions, 7))
portfolios.append(MultiPeriodPortfolio(name=f"Xgboost ifa MV"))
preselected_datas.append(PreselectedCloseData(ifa_predictions, 7))


print("Predictions done")

previous_weights = None
for train_idx, test_idx in tqdm(cv.split(X), 
                                desc="Generating LSTM portfolios", 
                                unit="portfolio",
                                total=cv.get_n_splits(X)):
    
    train_data = X.take(train_idx)
    test_data = X.take(test_idx)
    test_date = test_data.index[0]

    for index, preselected_data in enumerate(preselected_datas):
        preselected_train_data = preselected_data.filter_close_df(train_data, test_date)
        preselected_test_data = preselected_data.filter_close_df(test_data, test_date)
        model.fit(preselected_train_data)
        portfolio = model.predict(preselected_test_data)
        portfolios[index].append(portfolio)

population = Population(portfolios)
fig = population.plot_cumulative_returns()
show(fig)