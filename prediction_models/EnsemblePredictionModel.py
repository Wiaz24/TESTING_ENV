from keras.api.models import Model, load_model
from keras.api.layers import Input, Dense, LSTM, Dropout, GRU, SimpleRNN, Average, Reshape
from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping
from tqdm import tqdm
from .IPredictionModel import *
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

class EnsemblePredictionModel(IPredictionModel):
    @property
    def is_multi_model(self) -> bool:
        return False
    
    @property
    def file_extension(self) -> str:
        return ".keras"
    
    def __init__(self, tickers: list[str]):
        self._tickers = tickers
        self.model: Model = None

        # Parametry modelu
        self.history_length = 60
        self.hidden_nodes = 128  # Zgodnie z artykułem
        self.dropout_rate = 0.2  # Zgodnie z artykułem
        self.learning_rate = 0.0005  # Zgodnie z artykułem
        self.loss_function = 'mean_absolute_error'  # Możesz też użyć 'huber_loss'

        self.epochs = 100
        self.batch_size = 32  # Zgodnie z artykułem
        self.patience = 5  # Zmniejszono z 10 na 5, aby wcześniej zatrzymać uczenie

        self._init_ensemble_model()

    def _load_model(self, model_file: Path):
        self.model = load_model(model_file)

    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        features = FeaturesData(market_data)
        
        # Dodanie cechy 'medium' zgodnie z artykułem (średnia z high i low)
        features.add_feature(name='medium',
                          func=lambda df: (df['High'] + df['Low']) / 2)
        
        # Podstawowe cechy
        features.add_feature(name='open', func=lambda df: df['Open'])
        features.add_feature(name='high', func=lambda df: df['High'])
        features.add_feature(name='low', func=lambda df: df['Low'])
        features.add_feature(name='volume', func=lambda df: df['Volume'])
        
        # Zwroty logarytmiczne
        for i in range(1, 6):  # Ostatnie 5 dni
            features.add_feature(name=f'close_return_{i}',
                               func=lambda df, i=i: np.log(df['Close'] / df['Close'].shift(i)))
                               
            features.add_feature(name=f'medium_return_{i}',
                               func=lambda df, i=i: np.log(
                                   (df['High'] + df['Low']) / 2 / 
                                   ((df['High'].shift(i) + df['Low'].shift(i)) / 2)
                               ))
        
        # Dodanie cech technicznych
        features.add_feature(name='rsi_14', 
                          func=lambda df: df.ta.rsi(length=14, append=False))
        features.add_feature(name='momentum_10', 
                          func=lambda df: df.ta.mom(length=10, append=False))
        features.add_feature(name='atr_14', 
                          func=lambda df: df.ta.atr(length=14, append=False))
        
        # Zmiana - teraz przewidujemy przyszłą cenę zamknięcia, a nie zwrot logarytmiczny
        features.add_target(name='next_day_close',
                            func=lambda df: df['Close'].shift(-1))
        
        # Dodanie aktualnej ceny zamknięcia jako dodatkowy element do przewidywania zwrotu
        features.add_feature(name='current_close', 
                          func=lambda df: df['Close'])
        
        features.check_for_inf()
        features.check_for_nan()
        return features
    
    def _init_ensemble_model(self):
        # Warstwa wejściowa - przyjmuje wektor cech
        input_shape = Input(shape=(None,))
        
        # Warstwa reshape do 3D dla warstw rekurencyjnych
        # Dostosowuję kształt do wymogów naszego modelu
        reshaped = Reshape((-1, 1))(input_shape)
        
        # RNN layer z aktywacją tanh zgodnie z artykułem
        rnn_layer = SimpleRNN(self.hidden_nodes, 
                            activation='tanh',
                            return_sequences=False)(reshaped)
        
        # LSTM layer z aktywacją tanh zgodnie z artykułem
        lstm_layer = LSTM(self.hidden_nodes,
                         activation='tanh',
                         return_sequences=False)(reshaped)
        
        # GRU layer z aktywacją tanh zgodnie z artykułem
        gru_layer = GRU(self.hidden_nodes,
                       activation='tanh',
                       return_sequences=False)(reshaped)
        
        # Uśrednienie wyjść wszystkich warstw RNN zgodnie z artykułem
        average_layer = Average()([rnn_layer, lstm_layer, gru_layer])
        
        # Dropout zgodnie z artykułem
        dropout_layer = Dropout(self.dropout_rate)(average_layer)
        
        # Warstwa gęsta z aktywacją ReLU zgodnie z artykułem
        dense_layer = Dense(32, activation='relu')(dropout_layer)
        
        # Wyjście - liniowa warstwa dla predykcji ceny zamknięcia
        output_layer = Dense(1, activation='linear')(dense_layer)
        
        # Tworzenie modelu
        model = Model(inputs=input_shape, outputs=output_layer)
        
        # Kompilacja modelu z optymalizatorem SGD zgodnie z artykułem
        model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                     loss=self.loss_function)
        
        self.model = model
    
    def fit(self, features: FeaturesData, val_features: FeaturesData = None):
        X_all = []
        y_all = []

        for ticker in self.tickers:
            if ticker not in features.tickers:
                continue
            
            try:
                X_ticker_df = features.get_features_for_ticker(ticker)
                X_ticker = X_ticker_df.values
                y_ticker = features.get_target_for_ticker(ticker)

                # Dodajemy dane do list
                X_all.append(X_ticker)
                y_all.append(y_ticker)
            except Exception as e:
                print(f"Nie udało się przetworzyć danych treningowych dla {ticker}: {str(e)}")
        
        if not X_all:
            raise ValueError("Brak danych do trenowania")
        
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        
        # Przygotowanie danych walidacyjnych, jeśli są dostępne
        validation_data = None
        if val_features is not None:
            X_val = []
            y_val = []
            
            for ticker in self.tickers:
                if ticker not in val_features.tickers:
                    continue
                    
                try:
                    X_ticker_val = val_features.get_features_for_ticker(ticker).values
                    y_ticker_val = val_features.get_target_for_ticker(ticker).values
                    
                    X_val.append(X_ticker_val)
                    y_val.append(y_ticker_val)
                except Exception as e:
                    print(f"Nie udało się przetworzyć danych walidacyjnych dla {ticker}: {str(e)}")
            
            if X_val:
                X_val = np.vstack(X_val)
                y_val = np.concatenate(y_val)
                validation_data = (X_val, y_val)
                print(f"Przygotowano dane walidacyjne o kształcie: {X_val.shape}")
            else:
                print("Brak danych walidacyjnych dla żadnego tickera.")
        
        # Early stopping zgodnie z artykułem - monitorujemy 'val_loss' jeśli są dane walidacyjne
        monitor = 'val_loss' if validation_data is not None else 'loss'
        early_stopping = EarlyStopping(
            monitor=monitor, 
            patience=self.patience, 
            restore_best_weights=True,
            min_delta=0.0001,  # Dodano minimalną zmianę monitorowanej wartości
            verbose=1
        )
        
        # Przygotowanie dodatkowych callbacków, np. do wyświetlania postępu
        callbacks = [early_stopping]
            
        # Trenowanie modelu z wyświetlaniem walidacji
        history = self.model.fit(
            X_all, 
            y_all, 
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1, 
            shuffle=True,
            validation_data=validation_data
        )
        
        # Wypisanie finalnych wartości loss
        if validation_data is not None:
            print(f"Końcowy training loss: {history.history['loss'][-1]:.6f}")
            print(f"Końcowy validation loss: {history.history['val_loss'][-1]:.6f}")
        else:
            print(f"Końcowy training loss: {history.history['loss'][-1]:.6f}")
        
        return history
    
    def predict(self, features: FeaturesData, verbose: str | int = 1) -> PredictionsData:
        shifted_index = self._generate_shifted_index(features.df_index, 1)
        predictions = PredictionsData(_index = shifted_index, _tickers = features.tickers)
        
        # Generowanie predykcji dla każdego tickera
        for ticker in tqdm(self.tickers, desc="Predicting", unit="ticker", disable=not verbose):
            if ticker not in features.tickers:
                print(f"Ostrzeżenie: Brak danych dla tickera {ticker}")
                continue
                
            try:
                X_ticker = features.get_features_for_ticker(ticker).values
                y_ticker = features.get_target_for_ticker(ticker).values
                
                # Pobierz aktualną cenę zamknięcia dla wyliczenia zwrotu logarytmicznego
                current_close = features.get_features_for_ticker(ticker)['current_close'].values
                
                # Predykcja - bez zmiany kształtu - dostajemy przewidywane ceny zamknięcia
                predicted_close_prices = self.model.predict(X_ticker, verbose=0).flatten()
                
                # Konwersja przewidywanych cen zamknięcia na zwroty logarytmiczne
                # Wzór: log(predicted_close / current_close)
                predicted_log_returns = np.log(predicted_close_prices / current_close)
                
                # Zapisanie przewidywanych zwrotów logarytmicznych
                predictions_series = pd.Series(predicted_log_returns, index=shifted_index)
                predictions.add_prediction(ticker, predictions_series)
                
                # Konwersja rzeczywistych cen zamknięcia na zwroty logarytmiczne
                # Zwróć uwagę, że y_ticker zawiera rzeczywiste przyszłe ceny zamknięcia
                actual_log_returns = np.log(y_ticker.flatten() / current_close)
                
                # Dodanie rzeczywistych zwrotów logarytmicznych
                correct_data_series = pd.Series(actual_log_returns, index=shifted_index)
                predictions.add_correct_data(ticker, correct_data_series)
                
            except Exception as e:
                print(f"Błąd podczas przewidywania dla {ticker}: {str(e)}")
                # Utworzenie pustych danych dla zachowania spójności
                dummy_series = pd.Series(np.zeros(len(features.df_index)), index=features.df_index)
                
                try:
                    predictions.add_prediction(ticker, dummy_series)
                    predictions.add_correct_data(ticker, dummy_series)
                except Exception as inner_e:
                    print(f"Nie można dodać pustych danych dla {ticker}: {str(inner_e)}")
        
        return predictions
    
    def _save_model(self, model_file: Path):
        self.model.save(model_file)