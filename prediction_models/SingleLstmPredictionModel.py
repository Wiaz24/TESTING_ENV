import os
from keras.api.models import Model, load_model
from keras.api.layers import Input, Dense, LSTM, Dropout
from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping
from .IPredictionModel import *
import pandas as pd
import pandas_ta as ta
import numpy as np

class SingleLstmPredictionModel(IPredictionModel):

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.model: Model = None

        self.history_length = 60
        self.hidden_nodes = 10
        self.learning_rate = 0.001
        self.dropout = 0.1
        self.recurrent_dropout = 0.2
        self.loss_function = 'mean_absolute_error'

        self.epochs = 100
        self.batch_size = 100
        self.patience = 10

        self._init_lstm_model()

    def load_model(self, model_file: Path):
        if not isinstance(model_file, Path):
            model_file = Path(model_file)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} does not exist.")
        if not model_file.suffix == ".keras":
            raise ValueError(f"Model file {model_file} is not a .keras file.")
        self.model = load_model(model_file)

    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        features = FeaturesData(market_data)
       
        for i in range(self.history_length):
            features.add_feature(name=f'ln_close_return_{i}',
                                 func=lambda df: np.log(df['Close'].shift(i) / df['Close'].shift(i+1)))
            
        features.add_target(name='ln_close_return',
                             func=lambda df: np.log(df['Close'].shift(-1) / df['Close']))
        
        features.check_for_inf()
        features.check_for_nan()
        return features
    
    def _init_lstm_model(self):
        input_layer = Input(shape=(self.history_length, 1))
        
        lstm_layer = LSTM(self.hidden_nodes,
                        recurrent_dropout=self.recurrent_dropout, 
                        dropout=self.dropout,
                        activation='relu')(input_layer)
        
        output_layer = Dense(1, activation='linear')(lstm_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                    loss=self.loss_function)
        self.model = model
    
    def fit(self, features: FeaturesData):
        X_all = []
        y_all = []

        for ticker in self.tickers:
            if ticker not in features.tickers:
                raise ValueError(f"Data for ticker {ticker} not found.")
            # print(features.dataframes[ticker].shape)
            X_ticker_df = features.get_features_for_ticker(ticker)
            # print(X_ticker_df.shape)
            X_ticker = X_ticker_df.values
            y_ticker = features.get_target_for_ticker(ticker)

            # Reshape the data to 3D array
            X_ticker = np.reshape(X_ticker, (X_ticker.shape[0], X_ticker.shape[1], 1))
            X_all.append(X_ticker)
            y_all.append(y_ticker)
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        # Create an EarlyStopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)
            
            # Fit the model
        self.model.fit(X_all, 
                        y_all, 
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        callbacks=[early_stopping],
                        verbose=1, 
                        shuffle=True)
    
    def predict(self, features: FeaturesData, verbose: str = "auto") -> PredictionsData:
        predictions = PredictionsData(_index = features.df_index, _tickers = features.tickers)
        for ticker in self.tickers:
            X_ticker = features.get_features_for_ticker(ticker).values
            y_ticker = features.get_target_for_ticker(ticker).values
            # Reshape the data to 3D array
            X_ticker = np.reshape(X_ticker, (X_ticker.shape[0], X_ticker.shape[1], 1))

            # Make predictions
            predictions_series = pd.Series(self.model.predict(X_ticker, verbose=verbose).flatten(), index=features.df_index)
            predictions.add_prediction(ticker, predictions_series)
            # Add correct data

            correct_data_series = pd.Series(y_ticker.flatten(), index=features.df_index)
            predictions.add_correct_data(ticker, correct_data_series)
        return predictions
    
    def save_model(self, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_file = f"{model_path}/single_lstm_model.keras"
        self.model.save(model_file)
        print(f"Model saved to {model_file}")