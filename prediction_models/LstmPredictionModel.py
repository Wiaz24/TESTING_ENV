from keras.api.models import Model, load_model
from keras.api.layers import Input, Dense, LSTM, Dropout
from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping
from .IPredictionModel import *
import pandas as pd
import pandas_ta as ta
import numpy as np

class LstmPredictionModel(IPredictionModel):

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.models: dict[str, Model] = {}

        self.history_length = 60
        self.hidden_nodes = 10
        self.learning_rate = 0.001
        self.dropout = 0.1
        self.recurrent_dropout = 0.2
        self.loss_function = 'mean_squared_error'

        self.epochs = 100
        self.batch_size = 100
        self.early_stopping = EarlyStopping(monitor='loss', 
                                            patience=20, 
                                            restore_best_weights=True)

        self._init_lstm_models()

    def load_model(self, model_path: Path):
        for model_file in list(model_path.glob("*.keras")):
            ticker = model_file.stem
            if ticker in self.models:
                self.models[ticker] = load_model(model_file)
            else:
                raise ValueError(f"Model for ticker {ticker} not found in the loaded models.")
            
    def ohclv_to_features(self, ohclv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        features = {}
        for ticker in self.tickers:
            if ticker not in ohclv:
                raise ValueError(f"OHCLV data for ticker {ticker} not found.")
            df = ohclv[ticker].copy()
            indicators = pd.DataFrame(index=df.index)

            for i in range(self.history_length):
                # 1. ln(close_i / close_i-1)
                indicators[f'ln_close_return_{i}'] = np.log(df['Close'].shift(i) / df['Close'].shift(i+1))

            indicators['target_ln_close_return'] = np.log(df['Close'].shift(-1) / df['Close'])
            indicators.dropna(inplace=True)
            features[ticker] = indicators
        self._check_for_inf(features)
        return features
    
    def _init_lstm_models(self):
        for ticker in self.tickers:
            # Create and compile the LSTM model
            input_layer = Input(shape=(self.history_length, 1))
            
            lstm_layer = LSTM(self.hidden_nodes,
                            recurrent_dropout=self.recurrent_dropout, 
                            dropout=self.dropout)(input_layer)
            
            output_layer = Dense(1, activation='linear')(lstm_layer)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                        loss=self.loss_function)
            self.models[ticker] = model
    
    def fit(self, X: dict[str, pd.DataFrame], y: dict[str, pd.DataFrame]):
        for ticker in self.tickers:
            if ticker not in X or ticker not in y:
                raise ValueError(f"Data for ticker {ticker} not found.")
            X_ticker = X[ticker].values
            y_ticker = y[ticker].values

            # Reshape the data to 3D array
            X_ticker = np.reshape(X_ticker, (X_ticker.shape[0], X_ticker.shape[1], 1))

            # Create an EarlyStopping callback
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            
            # Fit the model
            self.models[ticker].fit(X_ticker, 
                                    y_ticker, 
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    callbacks=[early_stopping],
                                    verbose=1)
            
    def evaluate(self, X: dict[str, pd.DataFrame], y: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        predictions = {}
        for ticker in self.tickers:
            if ticker in X and ticker in y:
                y_pred = self.models[ticker].predict(X[ticker])
                df = pd.DataFrame(index=y[ticker].index)
                df['predicted_value'] = y_pred.astype(np.float64)
                df['actual_value'] = y[ticker].values
                predictions[ticker] = df
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
        self.print_metrics(predictions, 'predicted_value', 'actual_value')
        return predictions
    
    def predict(self, X: dict[str, pd.Series]) -> dict[str, float]:
        predictions = {}
        for ticker in self.tickers:
            if ticker in X:
                X_ticker = X[ticker].values
                # Reshape the data to 3D array
                X_ticker = np.reshape(X_ticker, (X_ticker.shape[0], X_ticker.shape[1], 1))
                predictions[ticker] = self.models[ticker].predict(X_ticker)
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
        return predictions
    
    def save_model(self, model_path: str):
        for ticker in self.tickers:
            model_file = Path(model_path) / f"{ticker}.keras"
            self.models[ticker].save(model_file)
            print(f"Model for {ticker} saved to {model_file}")