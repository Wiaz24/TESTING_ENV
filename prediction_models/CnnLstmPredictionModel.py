import os
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta

from keras.api.models import Model, Sequential, load_model
from keras.api.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, Input
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from tqdm import tqdm

from .IPredictionModel import IPredictionModel
from models.MarketData import MarketData
from models.FeaturesData import FeaturesData
from models.PredictionsData import PredictionsData

class CnnLstmPredictionModel(IPredictionModel):
    @property
    def is_multi_model(self) -> bool:
        return False
    
    @property
    def file_extension(self) -> str:
        return ".keras"

    def __init__(self, tickers: list[str]):
        self._tickers = tickers
        self.model = None
        
        # Hiperparametry CNN
        self.filters = 64
        self.kernel_size = 3
        self.pool_size = 1
        
        # Hiperparametry LSTM
        self.lstm_units = 50
        self.dropout_rate = 0.2
        
        # Ogólne hiperparametry uczenia
        self.epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.001
        self.patience = 10
        self.history_length = 10
        self.loss_function = 'mean_squared_error'

        self.features: dict[str, callable] = {
            'Close': lambda df: df['Close'],
            'SMA_12': lambda df: ta.sma(close=df['Close'], length=12),
            'EMA_14': lambda df: ta.ema(close=df['Close'], length=14),
            'RSI_14': lambda df: ta.rsi(close=df['Close'], length=14),
            'ROC_12': lambda df: ta.roc(close=df['Close'], length=12),
            'TR': lambda df: ta.true_range(high=df['High'], low=df['Low'], close=df['Close']),
            'ATR_14': lambda df: ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14),
            'MOM_10': lambda df: ta.mom(close=df['Close'], length=10),
            'CCI_20': lambda df: ta.cci(high=df['High'], low=df['Low'], close=df['Close'], length=20)
        }

        self.target = 'Next_day_close'

        # Inicjalizacja modelu
        self._init_cnn_lstm_model()
        
    def _init_cnn_lstm_model(self):

        input_shape = (self.history_length, len(self.features))
    
        inputs = Input(shape=input_shape)
        
        x = Conv1D(filters=self.filters, 
                   kernel_size=self.kernel_size, 
                   activation='relu')(inputs)
        x = MaxPooling1D(pool_size=self.pool_size)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(self.lstm_units)(x)
        x = Dense(self.lstm_units, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss=self.loss_function)
        
        self.model = model

    def _load_model(self, model_file: Path):
        self.model = load_model(model_file)
    
    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        features = FeaturesData(market_data)
        
        for i in range(self.history_length):
            for feature_name, feature_func in self.features.items():
                features.add_feature(name=f'{feature_name}_{i}',
                                     func=lambda df, feature_func=feature_func: feature_func(df).shift(i))
        features.add_target(name=self.target,
                          func=lambda df: df['Close'].shift(-1))
        features.check_for_nan()
        features.check_for_inf()
        return features
  
    def _feature_to_tensor(self, features_df: pd.DataFrame) -> np.ndarray:
        index_len = len(features_df)
        history_length = self.history_length
        feature_names = list(self.features.keys())
        feature_count = len(feature_names)
        
        # Przygotuj tensor o odpowiednich wymiarach
        tensor = np.zeros((index_len, history_length, feature_count))
        
        # Utwórz odpowiednią strukturę kolumn
        for i in range(history_length):
            columns_for_this_step = [f'{feature_name}_{i}' for feature_name in feature_names]
            tensor[:, i, :] = features_df[columns_for_this_step].values

        if np.isnan(tensor).any():
            print(f"NaN values found in features")
        if np.isinf(tensor).any():
            print(f"Inf values found in features")
        return tensor
        
    def fit(self, features: FeaturesData, val_features: FeaturesData = None):
        X_all = []
        y_all = []
        
        for ticker in tqdm(self.tickers, desc="Preparing training data", unit="ticker"):
            if ticker not in features.tickers:
                continue
            
            # Pobierz cechy i cel dla danego tickera
            X_ticker = features.get_features_for_ticker(ticker)
            y_ticker = features.get_target_for_ticker(ticker)
            
            # Przygotuj tensor dla tego tickera
            X_tensor = self._feature_to_tensor(X_ticker)
            
            # Rozszerz listy o poszczególne próbki (a nie cały tensor)
            X_all.extend(X_tensor)
            y_all.extend(y_ticker.values)
        
        # Konwersja na tablice numpy
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Przygotowanie danych walidacyjnych, jeśli zostały dostarczone
        validation_data = None
        if val_features is not None:
            X_val_all = []
            y_val_all = []
            
            for ticker in tqdm(self.tickers, desc="Preparing validation data", unit="ticker"):
                if ticker not in val_features.tickers:
                    continue
                    
                # Pobierz cechy i cel dla danego tickera
                X_val_ticker = val_features.get_features_for_ticker(ticker)
                y_val_ticker = val_features.get_target_for_ticker(ticker)
                
                # Przygotuj tensor dla tego tickera
                X_val_tensor = self._feature_to_tensor(X_val_ticker)
                
                # Rozszerz listy o poszczególne próbki
                X_val_all.extend(X_val_tensor)
                y_val_all.extend(y_val_ticker.values)
            
            if X_val_all:  # Upewniamy się, że mamy jakieś dane walidacyjne
                X_val_all = np.array(X_val_all)
                y_val_all = np.array(y_val_all)
                validation_data = (X_val_all, y_val_all)
        
        # Sprawdzenie czy mamy dane treningowe
        if len(X_all) == 0 or len(y_all) == 0:
            raise ValueError("No training data available after processing")
        
        # Wypisz informacje o kształcie danych
        print(f"Training data shape: X={X_all.shape}, y={y_all.shape}")
        if validation_data is not None:
            print(f"Validation data shape: X={validation_data[0].shape}, y={validation_data[1].shape}")
        
        monitor_metric = 'val_loss' if validation_data is not None else 'loss'
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Trenowanie modelu
        self.model.fit(
            X_all, y_all,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=1,
            shuffle=True
        )
    
    def predict(self, features: FeaturesData, verbose: str = "auto") -> PredictionsData:
        shifted_index = self._generate_shifted_index(features.df_index, 1)
        predictions = PredictionsData(_index=shifted_index, _tickers=features.tickers)
        
        for ticker in tqdm(self.tickers, desc="Predicting", unit="ticker"):
            if ticker not in features.tickers:
                print(f"Warning: Ticker {ticker} not found in the provided features")
                continue
                
            try:
                X_ticker = features.get_features_for_ticker(ticker)
                correct_prices_series = features.get_target_for_ticker(ticker)
                
                # Pobierz aktualną cenę zamknięcia ('Close_0')
                current_close_prices = X_ticker['Close_0']

                # Sprawdź, czy są jakieś braki danych
                if current_close_prices.isna().any():
                    print(f"NaN values found in current_close_prices for {ticker}: {current_close_prices[current_close_prices.isna()]}")
                
                if correct_prices_series.isna().any():
                    print(f"NaN values found in correct_prices_series for {ticker}: {correct_prices_series[correct_prices_series.isna()]}")

                # Przygotuj tensor dla tego tickera
                X_input = self._feature_to_tensor(X_ticker)
                
                # Przewiduj na podstawie tensora
                predicted_prices = self.model.predict(X_input, verbose=0 if verbose == "auto" else verbose)
                
                # Spłaszcz wyniki predykcji jeśli potrzeba
                if len(predicted_prices.shape) > 1 and predicted_prices.shape[1] == 1:
                    predicted_prices = predicted_prices.flatten()
                    
                # Konwersja na serię z odpowiednim indeksem
                predicted_prices_series = pd.Series(predicted_prices, index=X_ticker.index)
                
                # Sprawdź, czy są NaN w przewidywanych cenach
                if predicted_prices_series.isna().any():
                    print(f"NaN values found in predicted_prices_series for {ticker}: {predicted_prices_series[predicted_prices_series.isna()]}")

                # Oblicz logarytmiczny zwrot na podstawie aktualnej ceny i przewidywanej ceny
                predicted_log_returns = np.log(predicted_prices_series / current_close_prices)
                
                # Oblicz rzeczywisty logarytmiczny zwrot
                correct_log_returns = np.log(correct_prices_series / current_close_prices)
                
                # Sprawdź, czy są NaN w zwrotach
                if predicted_log_returns.isna().any():
                    print(f"NaN values in predicted_log_returns for {ticker}: {predicted_log_returns[predicted_log_returns.isna()]}")
                    # Zastąp wartości NaN średnią z reszty danych lub usuń te dni
                    predicted_log_returns = predicted_log_returns.fillna(predicted_log_returns.mean())
                    
                if correct_log_returns.isna().any():
                    print(f"NaN values in correct_log_returns for {ticker}: {correct_log_returns[correct_log_returns.isna()]}")
                    # Zastąp wartości NaN średnią z reszty danych lub usuń te dni
                    correct_log_returns = correct_log_returns.fillna(correct_log_returns.mean())

                # Zmień indeks na przesunięty indeks (reprezentujący jutrzejszy dzień)
                predicted_returns_with_future_index = pd.Series(predicted_log_returns.values, index=shifted_index)
                actual_returns_with_future_index = pd.Series(correct_log_returns.values, index=shifted_index)
                
                # Dodatkowe sprawdzenie NaN
                if predicted_returns_with_future_index.isna().any():
                    print(f"NaN still present in predicted_returns_with_future_index for {ticker}")
                    predicted_returns_with_future_index = predicted_returns_with_future_index.fillna(0)
                    
                if actual_returns_with_future_index.isna().any():
                    print(f"NaN still present in actual_returns_with_future_index for {ticker}")
                    actual_returns_with_future_index = actual_returns_with_future_index.fillna(0)

                predictions.add_prediction(ticker, predicted_returns_with_future_index)
                predictions.add_correct_data(ticker, actual_returns_with_future_index)
                
            except Exception as e:
                print(f"Error during prediction for {ticker}: {str(e)}")
                # Tworzenie pustych serii dla zachowania spójności
                dummy_series = pd.Series(0, index=shifted_index)
                predictions.add_prediction(ticker, dummy_series)
                predictions.add_correct_data(ticker, dummy_series)
        
        # Sprawdź, czy są NaN w końcowych predykcjach
        if predictions.check_for_nan:
            print("Warning: NaN values found in predictions after processing")
            # Dodatkowe sprawdzenie i naprawienie każdego DataFrame
            for ticker, df in predictions._dataframes.items():
                if df.isna().any().any():
                    print(f"Fixing NaN values for ticker {ticker}")
                    df.fillna(0, inplace=True)
        
        if predictions.check_for_inf:
            print("Warning: Inf values found in predictions")
            # Możesz również naprawić wartości Inf
            for ticker, df in predictions._dataframes.items():
                df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return predictions
        
    def _generate_shifted_index(self, index: pd.Index, shift: int) -> pd.Index:
        """
        Generuje przesunięty indeks dla przewidywanych zwrotów.
        
        Args:
            index: Oryginalny indeks
            shift: Ilość dni przesunięcia
            
        Returns:
            Przesunięty indeks
        """
        new_index = []
        for i in range(len(index)):
            if i + shift < len(index):
                new_index.append(index[i + shift])
            else:
                # Sprawdź ostatni przeskok o 3 dni w indeksie
                found_weekend_skip = False
                for j in range(min(5, len(index) - 1)):  # Sprawdź ostatnie 5 próbek
                    if j + 1 < len(index) and (index[-j-1] - index[-j-2]).days == 3:
                        found_weekend_skip = True
                        break
                
                # Ustal ostatnią datę w nowym indeksie
                if new_index:
                    last_date = new_index[-1]
                else:
                    last_date = index[-1]
                
                # Dodaj 3 dni jeśli znaleziono przeskok weekendowy w ostatnich 5 próbkach, w przeciwnym razie 1 dzień
                days_to_add = 3 if found_weekend_skip else 1
                next_date = last_date + pd.Timedelta(days=days_to_add)
                new_index.append(next_date)
        
        return pd.Index(new_index)
    
    def _save_model(self, model_file: Path, ticker: str = None):
        self.model.save(model_file)