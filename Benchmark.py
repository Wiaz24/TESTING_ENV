import pandas as pd
from pathlib import Path

from skfolio import MultiPeriodPortfolio, Population, RiskMeasure
from plotly.io import show
from tqdm import tqdm
from models.MarketData import MarketData
from models.PreselectedTickers import PreselectedTickers
from models.WorstCaseOmega import WorstCaseOmega
from models.PredictionBasedWorstCaseOmega import PredictionBasedWorstCaseOmega
from prediction_models.IPredictionModel import IPredictionModel
from prediction_models.SingleLstmPredictionModel import SingleLstmPredictionModel
from prediction_models.XgboostPredictionModel import XgboostPredictionModel
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction, InverseVolatility, EqualWeighted, Random
from skfolio.preprocessing import prices_to_returns

start_date = pd.Timestamp('2010-01-01')
split_date = pd.Timestamp('2018-01-04')
end_date = pd.Timestamp('2025-01-01')
market_data_dir = Path("data/tickers")

holding_period = 1  # obliczanie mv co 1 dzień
fitting_period = 60 # obliczanie mv na podstawie ostatnich 60 dni

market_data = MarketData(market_data_dir)
full_market_data = market_data.copy()
market_data.crop_data(split_date, end_date)

prediction_models: dict[IPredictionModel, Path] = {
    SingleLstmPredictionModel(market_data.tickers): Path("trained_models/lstm_mae/single_lstm_model.keras"),
    XgboostPredictionModel(market_data.tickers): Path("trained_models/xgboost_normalized")
}

X = prices_to_returns(market_data.close_df)
cv = WalkForward(train_size=fitting_period, test_size=holding_period)

portfolio_models: list[ObjectiveFunction] = [
    MeanRisk(risk_measure=RiskMeasure.VARIANCE),
    WorstCaseOmega(delta=0.8),
    InverseVolatility(),
    EqualWeighted(),
    Random()
]

population = Population([])

print("Creating all asset portfolios")
for pmodel in portfolio_models:
    mpportfolio = cross_val_predict(pmodel, X, cv=cv, n_jobs=-1)
    mpportfolio.name = f"All asstes - {pmodel.__class__.__name__}"
    population.append(mpportfolio)

print("Creating prediction based portfolios")
for pred_model, path in prediction_models.items():
    print(f"Creating portfolio for {pred_model.__class__.__name__}")
    pred_model.load_model(path)
    features = pred_model.market_to_features_data(full_market_data)
    features.crop_data(split_date, end_date)

    predictions = pred_model.predict(features)
    preselected_tickers = PreselectedTickers(predictions, 7)
    mpportfolios: dict[ObjectiveFunction, MultiPeriodPortfolio] = {}
    for port_model in portfolio_models:
        mpportfolio = MultiPeriodPortfolio(name=f"{pred_model.__class__.__name__} - {port_model.__class__.__name__}")
        mpportfolios[port_model] = mpportfolio

    pred_based_mpportfolio = MultiPeriodPortfolio(name=f"{pred_model.__class__.__name__} - Pred based worst case omega")
    pred_based_omega = PredictionBasedWorstCaseOmega()

    for train_idx, test_idx in tqdm(cv.split(X), 
                                    desc="Generating portfolios", 
                                    unit="portfolio",
                                    total=cv.get_n_splits(X)):
        train_data = X.take(train_idx)
        test_data = X.take(test_idx)
        test_date = test_data.index[0]

        preselected_train_data = preselected_tickers.filter_close_df(train_data, test_date)
        preselected_test_data = preselected_tickers.filter_close_df(test_data, test_date)

        for port_model, mpportfolio in mpportfolios.items():
            port_model.fit(preselected_train_data)
            portfolio = port_model.predict(preselected_test_data)
            mpportfolio.append(portfolio)

        preselected_predictions = predictions.predictions_df.loc[train_data.index]
        preselected_predictions = preselected_tickers.filter_close_df(preselected_predictions, test_date)
        pred_based_omega.fit(preselected_train_data, predictions=preselected_predictions)
        portfolio = pred_based_omega.predict(preselected_test_data)
        pred_based_mpportfolio.append(portfolio)

    for port_model, mpportfolio in mpportfolios.items():
        population.append(mpportfolio)
    population.append(pred_based_mpportfolio)

fig = population.plot_cumulative_returns()
show(fig)
