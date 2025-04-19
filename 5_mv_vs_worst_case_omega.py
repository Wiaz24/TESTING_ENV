from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from models.PreselectedTickers import PreselectedTickers
from models.MarketData import MarketData
from models.WorstCaseOmega import WorstCaseOmega
from models.PredictionBasedWorstCaseOmega import PredictionBasedWorstCaseOmega

from plotly.io import show
from pathlib import Path
from skfolio import MultiPeriodPortfolio, Population, Portfolio, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, BaseOptimization
from skfolio.preprocessing import prices_to_returns
import pandas as pd
from tqdm import tqdm

start_date = '2010-01-01'
split_date = '2020-01-01'
end_date = '2025-01-01'
market_data_dir = Path("data/tickers")

market_data = MarketData(market_data_dir)
model = SingleLstmPredictionModel(market_data.tickers)
model.load_model("trained_models/single_lstm/single_lstm_model.keras")
features = model.market_to_features_data(market_data)

market_data.crop_data(split_date, end_date)
features.crop_data(split_date, end_date)

prices = market_data.close_df
X = prices_to_returns(prices)

holding_period = 1  # obliczanie mv co 1 dzień
fitting_period = 60 # obliczanie mv na podstawie ostatnich 60 dni

cv = WalkForward(train_size=fitting_period, test_size=holding_period)

# transaction_cost = 0.001 / holding_period
transaction_cost = 0


print("Creating portfolios")
predictions = model.predict(features)
preselected_data = PreselectedTickers(predictions, 7)
# best_ticker = preselected_data._selected_tickers.iloc[0][0]
# # fig = predictions.plot_predictions(ticker=best_ticker)
# fig = preselected_data.plot_selection_histogram()
# show(fig)



models: list[BaseOptimization] = []
portfolios: list[MultiPeriodPortfolio] = []


portfolios.append(MultiPeriodPortfolio(name=f"Xgboost MV"))
models.append(MeanRisk(risk_measure=RiskMeasure.VARIANCE))

portfolios.append(MultiPeriodPortfolio(name=f"Xgboost Worst Case Omega"))
models.append(WorstCaseOmega(delta=0.8))

pred_based_mpp = MultiPeriodPortfolio(name=f"Xgboost Prediction based Worst Case Omega")
pred_based_model = PredictionBasedWorstCaseOmega()
print("Predictions done")

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

    for i, model in enumerate(models):
        model.fit(preselected_train_data)
        portfolio = model.predict(preselected_test_data)
        portfolios[i].append(portfolio)

    preselected_predictions = predictions.predictions_df.loc[train_data.index]
    preselected_predictions = preselected_data.filter_close_df(preselected_predictions, test_date)
    pred_based_model.fit(preselected_train_data, predictions=preselected_predictions)
    pred_based_portfolio = pred_based_model.predict(preselected_test_data)
    pred_based_mpp.append(pred_based_portfolio)

portfolios.append(pred_based_mpp)
population = Population(portfolios)
fig = population.plot_cumulative_returns()
show(fig)