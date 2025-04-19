from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from models.FeaturesData import FeaturesData
from models.MarketData import MarketData
from models.PredictionsData import PredictionsData

class IPredictionModel(ABC):
    @abstractmethod
    def __init__(self, tickers: list[str]):
        pass
        
    @abstractmethod
    def load_model(self, model_path: Path):
        pass

    @abstractmethod
    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        pass

    @abstractmethod
    def fit(self, features: FeaturesData):
        pass

    @abstractmethod
    def predict(self, features: FeaturesData) -> PredictionsData:
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        pass

    def _generate_shifted_index(self, index: pd.Index, shift: int) -> pd.Index:
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


        
