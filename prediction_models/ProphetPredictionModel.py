import itertools
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from tqdm import tqdm

from .IPredictionModel import IPredictionModel
from models.MarketData import MarketData
from models.FeaturesData import FeaturesData
from models.PredictionsData import PredictionsData

class ProphetPredictionModel(IPredictionModel):
    @property
    def is_multi_model(self) -> bool:
        return True

    @property
    def file_extension(self) -> str:
        return ".pkl"
    
    def __init__(self, tickers: list[str]):

        self._tickers = tickers
        self.models: dict[str, Prophet] = {}  # Słownik modeli Prophet dla każdego tickera
        
        # Parametry modelu
        self.forecast_periods = 1  # Ile dni w przód przewidujemy
        self.changepoint_prior_scale = 0.05  # Elastyczność trendu
        self.seasonality_prior_scale = 10.0  # Siła sezonowości
        self.holidays_prior_scale = 10.0  # Siła efektu świąt
        self.daily_seasonality = False  # Wyłączamy sezonowość dzienną, ponieważ dane są dzienne
        self.weekly_seasonality = True  # Włączamy sezonowość tygodniową
        self.yearly_seasonality = True  # Włączamy sezonowość roczną
        self.interval_width = 0.95  # Szerokość przedziału ufności dla prognoz

        self.params_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [15, 25, 35],
            'daily_seasonality': [False],
            'weekly_seasonality': [True],
            'yearly_seasonality': [True],
            'interval_width': [0.95]
        }

        self.best_params = None  # Najlepsze parametry wspólne dla wszystkich tickerów
        
        # Opcje dodatkowe
        self.use_log_returns = True  # Czy używać logarytmicznych zwrotów zamiast cen

    def _load_model(self, model_file: Path, ticker: str = None):
        with open(model_file, 'rb') as f:
            self.models[ticker] = pickle.load(f)
    
    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        features = FeaturesData(market_data)
        
        features.add_feature(name='log_return', 
                            func=lambda df: np.log(df['Close'] / df['Close'].shift(1)))
        features.add_feature(name='volume', func=lambda df: df['Volume'])
        features.add_feature(name='rsi_14', 
                          func=lambda df: df.ta.rsi(length=14, append=False))
        features.add_feature(name='momentum_10', 
                          func=lambda df: df.ta.mom(length=10, append=False))
        features.add_feature(name='atr_14', 
                          func=lambda df: df.ta.atr(length=14, append=False))
        features.add_target(name='next_day_log_return',
                            func=lambda df: np.log(df['Close'].shift(-1) / df['Close']))
        
        features.check_for_inf()
        features.check_for_nan()
        return features
    
    def _grid_search(self, train_df: pd.DataFrame):
        all_params = [dict(zip(self.params_grid.keys(), v)) for v in itertools.product(*self.params_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params)
            for feature in train_df.columns:
                if feature not in ['ds', 'y']:
                    m.add_regressor(feature)
            m.fit(train_df)
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        
        self.best_params = all_params[np.argmin(rmses)]
        print(self.best_params)


    def fit(self, features: FeaturesData, val_features: FeaturesData = None):

        for ticker in tqdm(self.tickers, desc="Training Prophet models", unit="ticker"):
            if ticker not in features.tickers:
                print(f"Warning: Ticker {ticker} not found in features")
                continue
            
            try:
                ticker_features = features.get_features_for_ticker(ticker)
                ticker_target = features.get_target_for_ticker(ticker)

                train_df = ticker_features.copy()
                train_df['ds'] = ticker_features.index
                train_df['y'] = ticker_target.values

                for feature in ticker_features.columns:
                    train_df[feature] = ticker_features[feature].values

                # Perform grid search if needed
                if self.best_params is None:
                    self._grid_search(train_df)
                
                model = Prophet(**self.best_params)
                for feature in ticker_features.columns:
                    model.add_regressor(feature)
                model.fit(train_df)
                self.models[ticker] = model
                
            except Exception as e:
                print(f"Error during training for {ticker}: {str(e)}")
    
    def predict(self, features: FeaturesData, verbose: str = "auto") -> PredictionsData:
        shifted_index = self._generate_shifted_index(features.df_index, 1)
        predictions = PredictionsData(_index=shifted_index, _tickers=features.tickers)
        
        # Dla każdego tickera wykonujemy predykcję
        for ticker in tqdm(self.tickers, desc="Predicting with Prophet", unit="ticker"):
            if ticker not in features.tickers:
                print(f"Warning: Ticker {ticker} not found in features")
                continue
            
            try:
                if ticker not in self.models:
                    print(f"Warning: No trained model for {ticker}")
                    continue
                
                ticker_features = features.get_features_for_ticker(ticker)
                ticker_target = features.get_target_for_ticker(ticker)

                model = self.models[ticker]
                
                test_df = ticker_features.copy()
                test_df['ds'] = ticker_features.index
                for feature in ticker_features.columns:
                    test_df[feature] = ticker_features[feature].values

                predictions_df = model.predict(test_df)

                predictions.add_prediction(ticker=ticker, 
                                            prediction=pd.Series(predictions_df['yhat'].values, index=shifted_index))
                predictions.add_correct_data(ticker=ticker,
                                            correct_data=pd.Series(ticker_target.values, index=shifted_index))

            except Exception as e:
                print(f"Error during predicting for {ticker}: {str(e)}")    
        
        return predictions
    
    def _save_model(self, model_file: Path, ticker: str = None):
        with open(model_file, 'wb') as f:
            pickle.dump(self.models[ticker], f)

# from pathlib import Path
# import os
# import numpy as np
# import pandas as pd
# import pickle
# from prophet import Prophet
# from tqdm import tqdm
# from joblib import Parallel, delayed
# import logging
# import itertools
# import warnings
# from prophet.diagnostics import cross_validation, performance_metrics

# from .IPredictionModel import IPredictionModel
# from models.MarketData import MarketData
# from models.FeaturesData import FeaturesData
# from models.PredictionsData import PredictionsData

# class ProphetPredictionModel(IPredictionModel):
#     @property
#     def is_multi_model(self) -> bool:
#         return True

#     @property
#     def file_extension(self) -> str:
#         return ".pkl"
    
#     def __init__(self, tickers: list[str], use_grid_search: bool = True, n_jobs: int = -1, verbose: bool = False):
#         """
#         Inicjalizuje model Prophet z opcjonalnym grid searchem.
        
#         Parameters:
#         -----------
#         tickers : list[str]
#             Lista symboli akcji
#         use_grid_search : bool
#             Czy używać grid search do znalezienia najlepszych parametrów
#         n_jobs : int
#             Liczba procesów do użycia podczas grid search (-1 oznacza wszystkie dostępne rdzenie)
#         verbose : bool
#             Czy wyświetlać szczegółowe logi
#         """
#         self._tickers = tickers
#         self.models: dict[str, Prophet] = {}  # Słownik modeli Prophet dla każdego tickera
#         self.best_params = None  # Najlepsze parametry wspólne dla wszystkich tickerów
#         self.verbose = verbose
        
#         # Parametry modelu (domyślne)
#         self.forecast_periods = 1  # Ile dni w przód przewidujemy
#         self.changepoint_prior_scale = 0.05  # Elastyczność trendu
#         self.seasonality_prior_scale = 10.0  # Siła sezonowości
#         self.holidays_prior_scale = 10.0  # Siła efektu świąt
#         self.seasonality_mode = 'additive'  # Tryb sezonowości
#         self.growth = 'linear'  # Typ wzrostu
#         self.n_changepoints = 25  # Liczba punktów zmiany
#         self.changepoint_range = 0.8  # Zakres danych, w którym szukane są punkty zmiany
#         self.daily_seasonality = False  # Wyłączamy sezonowość dzienną, ponieważ dane są dzienne
#         self.weekly_seasonality = True  # Włączamy sezonowość tygodniową
#         self.yearly_seasonality = True  # Włączamy sezonowość roczną
#         self.interval_width = 0.95  # Szerokość przedziału ufności dla prognoz
        
#         # Opcje dodatkowe
#         self.use_log_returns = True  # Czy używać logarytmicznych zwrotów zamiast cen
        
#         # Opcje grid search
#         self.use_grid_search = use_grid_search
#         self.n_jobs = n_jobs
#         self.cv_horizon = '30 days'  # Horyzont walidacji krzyżowej
#         self.cv_initial = '365 days'  # Początkowy rozmiar okna treningowego
#         self.cv_period = '90 days'  # Okresowość walidacji krzyżowej
#         self.cv_metric = 'rmse'  # Metryka do optymalizacji
        
#         # Siatka parametrów dla grid search
#         self.params_grid = {
#             'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
#             'seasonality_prior_scale': [0.1, 1.0, 10.0],
#             'holidays_prior_scale': [0.1, 1.0, 10.0],
#             'seasonality_mode': ['additive', 'multiplicative'],
#             'n_changepoints': [15, 25, 35],
#             'changepoint_range': [0.8, 0.9]
#         }
        
#         # Wyciszenie logów Prophet
#         self._disable_prophet_logs()
    
#     def _disable_prophet_logs(self):
#         """Wycisza wszystkie logi generowane przez Prophet i cmdstanpy."""
#         # Wyciszenie logów Prophet
#         logging.getLogger("cmdstanpy").disabled = True
    
#     def _load_model(self, model_file: Path, ticker: str = None):
#         with open(model_file, 'rb') as f:
#             self.models[ticker] = pickle.load(f)
    
#     def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
#         features = FeaturesData(market_data)
        
#         features.add_feature(name='log_return', 
#                             func=lambda df: np.log(df['Close'] / df['Close'].shift(1)))
#         features.add_feature(name='volume', func=lambda df: df['Volume'])
#         features.add_target(name='next_day_log_return',
#                             func=lambda df: np.log(df['Close'].shift(-1) / df['Close']))
        
#         features.check_for_inf()
#         features.check_for_nan()
#         return features
    
#     def _fit_single_prophet_model(self, ticker, ticker_features, ticker_target, params=None):
#         """
#         Dopasowuje pojedynczy model Prophet dla danego tickera.
        
#         Parameters:
#         -----------
#         ticker : str
#             Symbol akcji
#         ticker_features : DataFrame
#             Cechy dla danego tickera
#         ticker_target : Series
#             Wartości docelowe dla danego tickera
#         params : dict, optional
#             Parametry modelu, jeśli None to używane są domyślne parametry
            
#         Returns:
#         --------
#         Prophet model lub None w przypadku błędu
#         """
#         try:
#             if params is None:
#                 params = {
#                     'changepoint_prior_scale': self.changepoint_prior_scale,
#                     'seasonality_prior_scale': self.seasonality_prior_scale,
#                     'holidays_prior_scale': self.holidays_prior_scale,
#                     'seasonality_mode': self.seasonality_mode,
#                     'growth': self.growth,
#                     'n_changepoints': self.n_changepoints,
#                     'changepoint_range': self.changepoint_range,
#                     'daily_seasonality': self.daily_seasonality,
#                     'weekly_seasonality': self.weekly_seasonality,
#                     'yearly_seasonality': self.yearly_seasonality,
#                     'interval_width': self.interval_width
#                 }
            
#             model = Prophet(**params)
#             for feature in ticker_features.columns:
#                 model.add_regressor(feature)

#             train_df = pd.DataFrame({
#                 'ds': ticker_features.index,
#                 'y': ticker_target.values
#             })
            
#             for feature in ticker_features.columns:
#                 train_df[feature] = ticker_features[feature].values

#             model.fit(train_df)
#             return model
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error during training for {ticker}: {str(e)}")
#             return None
    
#     def _evaluate_params(self, params_dict, ticker_datasets):
      
#         metrics = []
        
#         for ticker, (features, target) in ticker_datasets.items():
#             try:
#                 # Tworzymy DataFrame w formacie wymaganym przez Prophet
#                 train_df = pd.DataFrame({
#                     'ds': features.index,
#                     'y': target.values
#                 })
                
#                 for feature in features.columns:
#                     train_df[feature] = features[feature].values
                
#                 model = Prophet(**params_dict)
#                 for feature in features.columns:
#                     model.add_regressor(feature)
                
#                 model.fit(train_df)
                
#                 df_cv = cross_validation(model, 
#                                       horizon=self.cv_horizon,
#                                       initial=self.cv_initial,
#                                       period=self.cv_period)
                
#                 df_p = performance_metrics(df_cv)
                
#                 # Dodajemy metrykę dla tego tickera
#                 metrics.append(df_p[self.cv_metric].mean())
#             except Exception as e:
#                 if self.verbose:
#                     print(f"Error during evaluation for {ticker}: {str(e)}")
#                 metrics.append(float('inf'))
        
#         # Zwracamy średnią metrykę dla wszystkich tickerów
#         valid_metrics = [m for m in metrics if m != float('inf')]
#         if not valid_metrics:
#             return float('inf')
#         return np.mean(valid_metrics)
    
#     def _grid_search(self, ticker_datasets):
#         """
#         Przeprowadza grid search wspólny dla wszystkich tickerów.
        
#         Parameters:
#         -----------
#         ticker_datasets : dict
#             Słownik {ticker: (features, target)}
            
#         Returns:
#         --------
#         dict: Najlepsze parametry
#         """
#         print(f"Performing grid search for all tickers...")
        
#         # Generujemy wszystkie kombinacje parametrów
#         keys, values = zip(*self.params_grid.items())
#         param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
#         # Dodajemy stałe parametry
#         for params in param_combinations:
#             params['daily_seasonality'] = self.daily_seasonality
#             params['weekly_seasonality'] = self.weekly_seasonality
#             params['yearly_seasonality'] = self.yearly_seasonality
#             params['interval_width'] = self.interval_width
#             params['growth'] = self.growth
        
#         # Równoległe przetwarzanie kombinacji parametrów
#         metrics = Parallel(n_jobs=self.n_jobs)(
#             delayed(self._evaluate_params)(params, ticker_datasets) for params in tqdm(param_combinations, desc="Evaluating parameters")
#         )
        
#         # Znajdujemy najlepsze parametry
#         best_idx = np.argmin(metrics)
#         best_params = param_combinations[best_idx]
#         best_metric = metrics[best_idx]
        
#         print(f"Best parameters for all tickers (avg {self.cv_metric}={best_metric:.4f}):")
#         for k, v in best_params.items():
#             print(f"  {k}: {v}")
        
#         return best_params
    
#     def fit(self, features: FeaturesData, val_features: FeaturesData = None):
#         """
#         Dopasowuje modele Prophet dla wszystkich tickerów, opcjonalnie używając grid search.
        
#         Parameters:
#         -----------
#         features : FeaturesData
#             Cechy treningowe
#         val_features : FeaturesData, optional
#             Cechy walidacyjne (nieużywane w Prophet, ale wymagane przez interfejs)
#         """
#         # Przygotowujemy dane dla wszystkich tickerów
#         ticker_datasets = {}
#         for ticker in self.tickers:
#             if ticker not in features.tickers:
#                 print(f"Warning: Ticker {ticker} not found in features")
#                 continue
            
#             ticker_features = features.get_features_for_ticker(ticker)
#             ticker_target = features.get_target_for_ticker(ticker)
#             ticker_datasets[ticker] = (ticker_features, ticker_target)
        
#         # Wykonujemy grid search, jeśli jest wymagany
#         if self.use_grid_search and ticker_datasets:
#             self.best_params = self._grid_search(ticker_datasets)
        
#         # Dopasowujemy modele dla wszystkich tickerów
#         for ticker in tqdm(self.tickers, desc="Training Prophet models", unit="ticker"):
#             if ticker not in ticker_datasets:
#                 continue
            
#             try:
#                 ticker_features, ticker_target = ticker_datasets[ticker]
                
#                 model = self._fit_single_prophet_model(
#                     ticker, ticker_features, ticker_target, 
#                     self.best_params if self.use_grid_search else None
#                 )
                
#                 if model is not None:
#                     self.models[ticker] = model
                
#             except Exception as e:
#                 if self.verbose:
#                     print(f"Error during training for {ticker}: {str(e)}")
    
#     def predict(self, features: FeaturesData, verbose: str = "auto") -> PredictionsData:
#         shifted_index = self._generate_shifted_index(features.df_index, 1)
#         predictions = PredictionsData(_index=shifted_index, _tickers=features.tickers)
        
#         # Dla każdego tickera wykonujemy predykcję
#         for ticker in tqdm(self.tickers, desc="Predicting with Prophet", unit="ticker"):
#             if ticker not in features.tickers:
#                 if self.verbose:
#                     print(f"Warning: Ticker {ticker} not found in features")
#                 continue
            
#             try:
#                 if ticker not in self.models:
#                     if self.verbose:
#                         print(f"Warning: No trained model for {ticker}")
#                     continue
                
#                 ticker_features = features.get_features_for_ticker(ticker)
#                 ticker_target = features.get_target_for_ticker(ticker)

#                 model = self.models[ticker]
                
#                 test_df = pd.DataFrame({'ds': ticker_features.index})
#                 for feature in ticker_features.columns:
#                     test_df[feature] = ticker_features[feature].values

#                 predictions_df = model.predict(test_df)

#                 predictions.add_prediction(ticker=ticker, 
#                                           prediction=pd.Series(predictions_df['yhat'].values, index=shifted_index))
#                 predictions.add_correct_data(ticker=ticker,
#                                            correct_data=pd.Series(ticker_target.values, index=shifted_index))

#             except Exception as e:
#                 if self.verbose:
#                     print(f"Error during predicting for {ticker}: {str(e)}")    
        
#         return predictions
    
#     def _save_model(self, model_file: Path, ticker: str = None):
#         with open(model_file, 'wb') as f:
#             pickle.dump(self.models[ticker], f)
            
#     def save_best_params(self, file_path: str):
#         """
#         Zapisuje najlepsze parametry do pliku JSON.
        
#         Parameters:
#         -----------
#         file_path : str
#             Ścieżka do pliku
#         """
#         import json
#         if self.best_params:
#             with open(file_path, 'w') as f:
#                 # Konwertujemy wartości na stringi, aby można było je zserializować
#                 serializable_params = {}
#                 for k, v in self.best_params.items():
#                     serializable_params[k] = str(v) if isinstance(v, (np.integer, np.floating)) else v
                
#                 json.dump(serializable_params, f, indent=4)
#         else:
#             print("No best parameters found. Grid search may not have been performed.")