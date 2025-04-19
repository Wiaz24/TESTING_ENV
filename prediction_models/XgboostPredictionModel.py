from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from .IPredictionModel import *
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import math
import random
import os
from typing import List, Dict, Tuple

class Firefly:
    def __init__(self, dimensions: int, min_values: List[float], max_values: List[float]):
        self.position = [random.uniform(min_values[i], max_values[i]) for i in range(dimensions)]
        self.fitness = float('inf')  # Minimalizacja
        self.brightness = 0.0  # Im niższa wartość funkcji, tym większa jasność
        
    def update_fitness(self, fitness_value: float):
        self.fitness = fitness_value
        self.brightness = 1.0 / (fitness_value + 1e-10)  # Unikamy dzielenia przez zero

class XgboostPredictionModel(IPredictionModel):

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.global_best_params = None
        self.param_ranges = {
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 1),
            'gamma': (0, 1),
            'min_child_weight': (1, 10)
        }

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

    def _perform_ifa_optimization(self, features: FeaturesData, samples_per_ticker: int = 100, 
                                  population_size: int = 30, max_iterations: int = 50):
        """
        Optymalizacja hiperparametrów XGBoost przy użyciu Improved Firefly Algorithm (IFA)
        """
        X_samples = []
        y_samples = []

        for ticker in self.tickers:
            if ticker in features.tickers:
                random_samples = np.random.choice(features.df_index, samples_per_ticker, replace=False)
                X_samples.append(features.get_features_for_ticker(ticker).loc[random_samples])
                y_samples.append(features.get_target_for_ticker(ticker).loc[random_samples])
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
            
        X_samples = pd.concat(X_samples)
        y_samples = pd.concat(y_samples)

        print(f"Łączna liczba próbek do IFA: {len(X_samples)}")
        
        # Konwertujemy y_samples do numpy array
        y_samples_values = y_samples.values.flatten()
        
        # Dzielimy dane na zbiór treningowy i walidacyjny, aby lepiej ocenić jakość modelu
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_samples, y_samples_values, test_size=0.3, random_state=42
        )
        
        # Definicja parametrów algorytmu IFA
        dimensions = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())
        min_values = [self.param_ranges[param][0] for param in param_names]
        max_values = [self.param_ranges[param][1] for param in param_names]
        
        # Zapewniamy większą różnorodność początkowej populacji
        fireflies = []
        for i in range(population_size):
            # Tworzenie zróżnicowanych początkowych pozycji
            firefly = Firefly(dimensions, min_values, max_values)
            # Losujemy pozycje z większą wariancją
            for d in range(dimensions):
                range_size = max_values[d] - min_values[d]
                # Dodajemy losowy szum do pozycji
                noise = np.random.normal(0, range_size * 0.3)
                firefly.position[d] = min(max_values[d], max(min_values[d], 
                                      min_values[d] + (i/population_size) * range_size + noise))
            fireflies.append(firefly)
        
        # Parametry Firefly Algorithm
        beta0 = 1.0  # Zwiększam początkową atrakcyjność
        gamma = 0.1  # Zwiększam współczynnik absorpcji światła dla lepszego lokalnego przeszukiwania
        alpha0 = 0.5  # Zwiększam początkową wartość współczynnika kroku
        
        # Inicjalizuj najlepszego świetlika z pierwszym świetlikiem
        best_firefly = Firefly(dimensions, min_values, max_values)
        best_firefly.position = fireflies[0].position.copy()
        best_firefly.fitness = float('inf')
        
        # Obliczenie wartości funkcji dopasowania dla każdego świetlika
        for i, firefly in enumerate(fireflies):
            # Konwersja pozycji świetlika na wartości parametrów
            params = {}
            for j, param_name in enumerate(param_names):
                if param_name == 'n_estimators' or param_name == 'min_child_weight':
                    params[param_name] = max(1, int(firefly.position[j]))
                else:
                    params[param_name] = firefly.position[j]
            
            # Tworzenie i trenowanie modelu z danymi parametrami
            model = xgb.XGBRegressor(
                objective='reg:squarederror', 
                random_state=42,
                **params
            )
            
            try:
                # Trenujemy model bez dodatkowych parametrów
                model.fit(X_train, y_train)
                
                # Używamy predykcji na zbiorze walidacyjnym do oceny
                y_pred = model.predict(X_valid)
                
                # Obliczamy MSE używając numpy array
                mse = np.mean((y_valid - y_pred) ** 2)
                
                # Dodajemy regularyzację do funkcji straty, żeby zachęcić do prostszych modeli
                reg_penalty = (params.get('reg_alpha', 0) + params.get('reg_lambda', 0)) * 0.01
                complexity_penalty = params.get('n_estimators', 100) * 0.0001
                mse = mse + reg_penalty + complexity_penalty
                
                firefly.update_fitness(mse)
                
                # Aktualizacja najlepszego znalezionego świetlika
                if firefly.fitness < best_firefly.fitness:
                    best_firefly.position = firefly.position.copy()
                    best_firefly.update_fitness(firefly.fitness)
                    
                print(f"Świetlik {i+1}/{population_size}, MSE: {mse:.6f}, params: {params}")
            except Exception as e:
                firefly.update_fitness(float('inf'))
                print(f"Błąd dla świetlika {i+1}: {str(e)}")
        
        # Główna pętla algorytmu IFA
        for t in range(max_iterations):
            print(f"\nIteracja {t+1}/{max_iterations}")
            
            # Obliczenie dynamicznego progu podziału na elitarne i zwykłe świetliki
            threshold = max(1, math.floor(population_size * math.exp(t / max_iterations - 1)))
            
            # Sortowanie świetlików według jasności (odwrotnie proporcjonalnej do fitness)
            fireflies.sort(key=lambda x: x.fitness)
            
            # Upewnijmy się, że mamy najlepszego świetlika
            if fireflies[0].fitness < best_firefly.fitness:
                best_firefly.position = fireflies[0].position.copy()
                best_firefly.update_fitness(fireflies[0].fitness)
            
            # Wprowadzamy różnorodność co kilka iteracji, aby uniknąć lokalnych minimów
            if t % 5 == 0 and t > 0:
                print("Wprowadzam różnorodność...")
                # Resetujemy połowę najgorszych świetlików
                for i in range(population_size // 2, population_size):
                    for d in range(dimensions):
                        # Całkowicie nowa losowa pozycja
                        fireflies[i].position[d] = min_values[d] + random.random() * (max_values[d] - min_values[d])
                    # Obliczamy na nowo fitness
                    params = {}
                    for j, param_name in enumerate(param_names):
                        if param_name == 'n_estimators' or param_name == 'min_child_weight':
                            params[param_name] = max(1, int(fireflies[i].position[j]))
                        else:
                            params[param_name] = fireflies[i].position[j]
                    
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror', 
                        random_state=42,
                        **params
                    )
                    
                    try:
                        # Trenujemy model bez dodatkowych parametrów
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_valid)
                        mse = np.mean((y_valid - y_pred) ** 2)
                        
                        reg_penalty = (params.get('reg_alpha', 0) + params.get('reg_lambda', 0)) * 0.01
                        complexity_penalty = params.get('n_estimators', 100) * 0.0001
                        mse = mse + reg_penalty + complexity_penalty
                        
                        fireflies[i].update_fitness(mse)
                        
                        if fireflies[i].fitness < best_firefly.fitness:
                            best_firefly.position = fireflies[i].position.copy()
                            best_firefly.update_fitness(fireflies[i].fitness)
                    except Exception as e:
                        fireflies[i].update_fitness(float('inf'))
            
            # Aktualizacja pozycji elitarnych świetlików (strategia przeszukiwania chaotycznego)
            for i in range(threshold):
                for d in range(dimensions):
                    # Generujemy losową wartość dla funkcji chaotycznej
                    c2 = random.random()
                    # Parametr kontroli stanu chaotycznego
                    mu = 3.8  # Zmniejszam lekko, aby zwiększyć eksplorację
                    # Obliczamy nową wartość chaotyczną
                    c1 = mu * c2 * (1 - c2)
                    
                    # Aktualizujemy pozycję w wymiarze d - używamy większej losowości
                    range_width = max_values[d] - min_values[d]
                    chaotic_pos = min_values[d] + c1 * range_width
                    current_pos = fireflies[i].position[d]
                    
                    # Z prawdopodobieństwem 0.7 używamy pozycji chaotycznej, inaczej zostajemy w miejscu
                    if random.random() < 0.7:
                        new_pos = chaotic_pos
                    else:
                        # Dodajemy niewielki losowy ruch
                        new_pos = current_pos + random.gauss(0, range_width * 0.1)
                    
                    # Ograniczamy pozycję do dozwolonego zakresu
                    fireflies[i].position[d] = max(min_values[d], min(max_values[d], new_pos))
                
                # Obliczenie nowej wartości fitness
                params = {}
                for j, param_name in enumerate(param_names):
                    if param_name == 'n_estimators' or param_name == 'min_child_weight':
                        params[param_name] = max(1, int(fireflies[i].position[j]))
                    else:
                        params[param_name] = fireflies[i].position[j]
                
                model = xgb.XGBRegressor(
                    objective='reg:squarederror', 
                    random_state=42,
                    **params
                )
                
                try:
                    # Trenujemy model bez dodatkowych parametrów
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_valid)
                    mse = np.mean((y_valid - y_pred) ** 2)
                    
                    reg_penalty = (params.get('reg_alpha', 0) + params.get('reg_lambda', 0)) * 0.01
                    complexity_penalty = params.get('n_estimators', 100) * 0.0001
                    mse = mse + reg_penalty + complexity_penalty
                    
                    prev_fitness = fireflies[i].fitness
                    fireflies[i].update_fitness(mse)
                    
                    if fireflies[i].fitness < best_firefly.fitness:
                        best_firefly.position = fireflies[i].position.copy()
                        best_firefly.update_fitness(fireflies[i].fitness)
                        
                    print(f"Elitarny świetlik {i+1}/{threshold}, MSE: {mse:.6f}, Poprawa: {prev_fitness - mse:.6f}")
                except Exception as e:
                    print(f"Błąd dla elitarnego świetlika {i+1}: {str(e)}")
            
            # Aktualizacja pozycji zwykłych świetlików (ruch w kierunku jaśniejszych i najlepszego)
            for i in range(threshold, population_size):
                moved = False
                for j in range(threshold):  # Sprawdzamy tylko elitarne świetliki jako potencjalnie jaśniejsze
                    # Jeśli świetlik j jest jaśniejszy od świetlika i
                    if fireflies[j].brightness > fireflies[i].brightness:
                        moved = True
                        # Obliczenie odległości między świetlikami
                        r = sum([(fireflies[i].position[d] - fireflies[j].position[d]) ** 2 
                                for d in range(dimensions)]) ** 0.5
                        
                        # Obliczenie atrakcyjności zależnej od odległości
                        beta = beta0 * math.exp(-gamma * r * r)
                        
                        # Współczynnik dla ruchu w kierunku najlepszego świetlika
                        omega = alpha0 * (1 - t / max_iterations)
                        
                        for d in range(dimensions):
                            # Ruch w kierunku jaśniejszego świetlika
                            delta1 = beta * (fireflies[j].position[d] - fireflies[i].position[d])
                            
                            # Ruch w kierunku najlepszego świetlika (strategia PSO)
                            delta2 = omega * (random.random() - 0.5) * (best_firefly.position[d] - fireflies[i].position[d])
                            
                            # Dodajemy losowy szum, aby zwiększyć eksplorację
                            random_noise = (random.random() - 0.5) * 0.1 * (max_values[d] - min_values[d])
                            
                            # Aktualizacja pozycji
                            fireflies[i].position[d] += delta1 + delta2 + random_noise
                            
                            # Ograniczenie pozycji do dozwolonego zakresu
                            fireflies[i].position[d] = max(min_values[d], min(max_values[d], fireflies[i].position[d]))
                
                # Jeśli nie było ruchu (nie było jaśniejszych świetlików), dodajemy losowy ruch
                if not moved:
                    for d in range(dimensions):
                        # Losowy ruch w kierunku najlepszego świetlika
                        omega = alpha0 * (1 - t / max_iterations)
                        delta = omega * (random.random() - 0.5) * (best_firefly.position[d] - fireflies[i].position[d])
                        fireflies[i].position[d] += delta
                        fireflies[i].position[d] = max(min_values[d], min(max_values[d], fireflies[i].position[d]))
                
                # Obliczenie nowej wartości fitness
                params = {}
                for j, param_name in enumerate(param_names):
                    if param_name == 'n_estimators' or param_name == 'min_child_weight':
                        params[param_name] = max(1, int(fireflies[i].position[j]))
                    else:
                        params[param_name] = fireflies[i].position[j]
                
                model = xgb.XGBRegressor(
                    objective='reg:squarederror', 
                    random_state=42,
                    **params
                )
                
                try:
                    # Trenujemy model bez dodatkowych parametrów
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_valid)
                    mse = np.mean((y_valid - y_pred) ** 2)
                    
                    reg_penalty = (params.get('reg_alpha', 0) + params.get('reg_lambda', 0)) * 0.01
                    complexity_penalty = params.get('n_estimators', 100) * 0.0001
                    mse = mse + reg_penalty + complexity_penalty
                    
                    prev_fitness = fireflies[i].fitness
                    fireflies[i].update_fitness(mse)
                    
                    if fireflies[i].fitness < best_firefly.fitness:
                        best_firefly.position = fireflies[i].position.copy()
                        best_firefly.update_fitness(fireflies[i].fitness)
                        
                    print(f"Zwykły świetlik {i+1-threshold}/{population_size-threshold}, MSE: {mse:.6f}, Poprawa: {prev_fitness - mse:.6f}")
                except Exception as e:
                    print(f"Błąd dla zwykłego świetlika {i+1-threshold}: {str(e)}")
            
            print(f"Najlepszy wynik w iteracji {t+1}: {best_firefly.fitness:.6f}")
        
        # Konwersja najlepszej pozycji na wartości parametrów
        best_params = {}
        for i, param_name in enumerate(param_names):
            if param_name == 'n_estimators' or param_name == 'min_child_weight':
                best_params[param_name] = max(1, int(best_firefly.position[i]))
            else:
                best_params[param_name] = best_firefly.position[i]
        
        print(f"\nNajlepsze znalezione parametry: {best_params}")
        print(f"Najlepsza wartość MSE: {best_firefly.fitness:.6f}")
        
        self.global_best_params = best_params
        return best_params

    def fit(self, features: FeaturesData, use_ifa=False):
        """
        Trenowanie modelu XGBoost dla każdego tickera
        
        Parameters:
        -----------
        features : FeaturesData
            Dane wejściowe i wyjściowe
        use_ifa : bool, optional (default=False)
            Jeśli True, używa Improved Firefly Algorithm do optymalizacji hiperparametrów
            Jeśli False, używa Grid Search
        """
        if use_ifa:
            self._perform_ifa_optimization(features)
        else:
            self._perform_global_grid_search(features)
            
        for ticker in self.tickers:
            if ticker in features.tickers:
                # Jeśli znaleziono globalne optymalne parametry, używamy ich
                if self.global_best_params:
                    self.models[ticker] = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        **self.global_best_params
                    )
                
                # Trenujemy model na danych dla danego tickera
                self.models[ticker].fit(features.get_features_for_ticker(ticker), 
                                        features.get_target_for_ticker(ticker))                         
            else:
                raise ValueError(f"Data for ticker {ticker} not found in the provided datasets.")
            
    def predict(self, features: FeaturesData) -> PredictionsData:
        # Tworzymy nowy indeks przesunięty o 1 próbkę w przód
        shifted_index = self._generate_shifted_index(features.df_index, 1)
        
        predictions = PredictionsData(_index = shifted_index, _tickers = features.tickers)
        for ticker in self.tickers:
            if ticker not in features.tickers:
                print(f"Ostrzeżenie: Brak danych dla tickera {ticker}")
                continue
                
            try:
                X_ticker = features.get_features_for_ticker(ticker)
                y_ticker = features.get_target_for_ticker(ticker)

                # Make predictions
                predictions_raw = self.models[ticker].predict(X_ticker)
                predictions_series = pd.Series(predictions_raw, index=shifted_index)
                predictions.add_prediction(ticker, predictions_series)
                
                correct_data_series = pd.Series(y_ticker.values.flatten(), index=shifted_index)
                predictions.add_correct_data(ticker, correct_data_series)

            except Exception as e:
                print(f"Błąd podczas przewidywania dla {ticker}: {str(e)}")
                # Create dummy data to maintain consistency
                dummy_series = pd.Series(np.zeros(len(features.df_index)), index=features.df_index)
                try:
                    predictions.add_prediction(ticker, dummy_series)
                    predictions.add_correct_data(ticker, dummy_series)
                except Exception as inner_e:
                    print(f"Nie można dodać pustych danych dla {ticker}: {str(inner_e)}")

        return predictions
    
    def save_model(self, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        for ticker in self.tickers:
            model_file = Path(model_path) / f"{ticker}.json"
            self.models[ticker].save_model(model_file)
            print(f"Model for {ticker} saved to {model_file}")