from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from .IPredictionModel import *
import pandas as pd
import pandas_ta as ta
import numpy as np

class XgboostPredictionModel(IPredictionModel):

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.global_best_params = None

        for ticker in self.tickers:
            # Initialize the model for each ticker
            self.models[ticker] = xgb.XGBRegressor()
            self.models[ticker].set_params(objective='reg:squarederror')

    def load_model(self, model_path: Path):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        for model_file in list(model_path.glob("*.json")):
            ticker = model_file.stem
            if ticker in self.models:
                self.models[ticker] = xgb.XGBRegressor()
                self.models[ticker].load_model(model_file)
            else:
                raise ValueError(f"Model for ticker {ticker} not found in the loaded models.")   
    
    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        features = FeaturesData(market_data)
        
        features.add_feature(name='ln_close_return_1',
                             func=lambda df: np.log(df['Close'] / df['Close'].shift(1)))
        features.add_feature(name='ln_close_return_2',
                             func=lambda df: np.log(df['Close'].shift(1) / df['Close'].shift(2)))
        features.add_feature(name='ln_close_return_3',
                             func=lambda df: np.log(df['Close'].shift(2) / df['Close'].shift(3)))
        features.add_feature(name='ln_close_return_4',
                             func=lambda df: np.log(df['Close'].shift(3) / df['Close'].shift(4)))
        
        features.add_feature(name='ln_high_open_ratio_0',
                                func=lambda df: np.log(df['High'] / df['Open']))
        features.add_feature(name='ln_high_open_ratio_1',
                                func=lambda df: np.log(df['High'] / df['Open'].shift(1)))
        features.add_feature(name='ln_high_open_ratio_2',
                                func=lambda df: np.log(df['High'] / df['Open'].shift(2)))
        features.add_feature(name='ln_high_open_ratio_3',
                                func=lambda df: np.log(df['High'] / df['Open'].shift(3)))
        
        features.add_feature(name='ln_high_open_ratio_lag1',
                                func=lambda df: np.log(df['High'].shift(1) / df['Open'].shift(1)))
        features.add_feature(name='ln_high_open_ratio_lag2',
                                func=lambda df: np.log(df['High'].shift(2) / df['Open'].shift(2)))
        features.add_feature(name='ln_high_open_ratio_lag3',
                                func=lambda df: np.log(df['High'].shift(3) / df['Open'].shift(3)))
        
        features.add_feature(name='true_range',
                                func=lambda df: pd.concat([df['High'] - df['Low'],
                                                          abs(df['High'] - df['Close'].shift(1)),
                                                          abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1))
        features.add_feature(name='atr_14',
                                func=lambda df: df.ta.atr(length=14))
        features.add_feature(name='momentum_10',
                                func=lambda df: df.ta.mom(length=10))
        features.add_feature(name='rsi_14',
                                func=lambda df: df.ta.rsi(length=14))
        

        features.add_target(name='next_day_log_return',
                                func=lambda df: np.log(df['Close'].shift(-1) / df['Close']))

        return features
    
    def _perform_global_grid_search(self, features: FeaturesData, samples_per_ticker: int = 100):
        X_samples = []
        y_samples = []

        for ticker in self.tickers:
            if ticker in features.tickers:
                random_samples = np.random.choice(features.df_index, samples_per_ticker, replace=False)
                X_samples.append(features.get_features_for_ticker(ticker).loc[random_samples])
                y_samples.append(features.get_targets_for_ticker(ticker).loc[random_samples])
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
            
        X_samples = pd.concat(X_samples)
        y_samples = pd.concat(y_samples)

        print(f"Łączna liczba próbek do Grid Search: {len(X_samples)}")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'subsample': [1],
            'learning_rate': [0.05, 0.06, 0.07],
            'reg_alpha': [0, 0.1, 0.2],
            'reg_lambda': [0, 0.1, 0.2],
            'colsample_bytree': [1],
            'colsample_bylevel': [1],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1]
        }

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )

        try:
            # Wykonaj Grid Search z 5-krotną walidacją krzyżową
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=5,
                verbose=1,
                n_jobs=-1
            )
        
            grid_search.fit(X_samples, y_samples)
            self.global_best_params = grid_search.best_params_ 
            
        except Exception as e:
            print(f"Błąd podczas globalnego Grid Search: {e}")
            return None
        
    def fit(self, features: FeaturesData):
        self._perform_global_grid_search(features)
        for ticker in self.tickers:
            if ticker in features.tickers:
                self.models[ticker].fit(features.get_features_for_ticker(ticker), 
                                        features.get_targets_for_ticker(ticker))                         
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
            
    def predict(self, features: FeaturesData) -> PredictionsData:
        predictions = PredictionsData(_index = features.df_index, _tickers = features.tickers)
        for ticker in self.tickers:
            X_ticker = features.get_features_for_ticker(ticker)
            y_ticker = features.get_targets_for_ticker(ticker)

            # Make predictions
            predictions_series = pd.Series(self.models[ticker].predict(X_ticker), index=features.df_index)
            predictions.add_prediction(ticker, predictions_series)
            
            # Add correct data
            correct_data_series = pd.Series(y_ticker.values.flatten(), index=features.df_index)
            predictions.add_correct_data(ticker, correct_data_series)

        return predictions
    
    def save_model(self, model_path: str):
        for ticker in self.tickers:
            model_file = Path(model_path) / f"{ticker}.json"
            self.models[ticker].save_model(model_file)
            print(f"Model for {ticker} saved to {model_file}")