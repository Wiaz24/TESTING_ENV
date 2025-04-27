from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from prediction_models.EnsemblePredictionModel import EnsemblePredictionModel
from prediction_models.CnnLstmPredictionModel import CnnLstmPredictionModel
from prediction_models.ProphetPredictionModel import ProphetPredictionModel
from models.PreselectedTickers import PreselectedTickers
from models.MarketData import MarketData

from plotly.io import show
from pathlib import Path
from skfolio.model_selection import WalkForward
from skfolio.preprocessing import prices_to_returns

start_date = '2010-01-01'
split_date = '2020-01-01'
end_date = '2025-01-01'
market_data_dir = Path("data/tickers")

market_data = MarketData(market_data_dir)
# model = XgboostPredictionModel(market_data.tickers)
# model = SingleLstmPredictionModel(market_data.tickers)
# model = EnsemblePredictionModel(market_data.tickers)
# model = CnnLstmPredictionModel(market_data.tickers)
model = ProphetPredictionModel(market_data.tickers)
model.load_model("trained_models")
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
best_ticker = preselected_data._selected_tickers.iloc[0][0]
fig = predictions.plot_predictions(ticker=best_ticker)
# fig = preselected_data.plot_selection_histogram()
show(fig)