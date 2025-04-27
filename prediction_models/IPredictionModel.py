from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from models.FeaturesData import FeaturesData
from models.MarketData import MarketData
from models.PredictionsData import PredictionsData
from typeguard import typechecked

class IPredictionModel(ABC):
    _tickers: list[str] = None

    @property
    def tickers(self) -> list[str]:
        return self._tickers
    
    @property
    @abstractmethod
    def is_multi_model(self) -> bool:
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        pass

    @abstractmethod
    def __init__(self, tickers: list[str]):
        pass
        
    @typechecked
    def load_model(self, common_path: Path | str):
        if isinstance(common_path, str):
            common_path = Path(common_path)

        # Single model
        if not self.is_multi_model:
            model_file = common_path / self.__class__.__name__ / f"model{self.file_extension}"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file {model_file} does not exist.")
            self._load_model(model_file)
        
        # Multi model
        else:
            common_path = common_path / self.__class__.__name__
            model_files = list(common_path.glob(f"*{self.file_extension}"))
            if not model_files:
                raise FileNotFoundError(f"Model files for {self.__class__.__name__} do not exist.")
            for ticker in self.tickers:
                model_file = next((f for f in model_files if ticker in f.name), None)
                if not model_file:
                    raise FileNotFoundError(f"Model file for {ticker} does not exist.")
                self._load_model(model_file, ticker)

    @typechecked
    @abstractmethod
    def _load_model(self, model_file: Path, ticker: str = None):
        """Implements loading logic for specified model type."""
        pass
    
    @typechecked
    @abstractmethod
    def market_to_features_data(self, market_data: MarketData) -> FeaturesData:
        pass

    @typechecked
    @abstractmethod
    def fit(self, features: FeaturesData, val_features: FeaturesData = None): 
        pass

    @typechecked
    @abstractmethod
    def predict(self, features: FeaturesData, verbose: str | int = 1) -> PredictionsData:
        pass

    @typechecked
    def save_model(self, common_path: Path | str):
        """
        Saves the model to the specified path.
        Model is saved in the format: 
        - single model: common_path/model_class/model.model_extension
        - multi model: common_path/model_class/ticker.model_extension
        """
        if isinstance(common_path, str):
            common_path = Path(common_path)
        
        # Single model
        if not self.is_multi_model:
            model_file = common_path / self.__class__.__name__ / f"model{self.file_extension}"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_model(model_file)

        # Multi model
        else:
            common_path = common_path / self.__class__.__name__
            common_path.mkdir(parents=True, exist_ok=True)
            for ticker in self.tickers:
                model_file = common_path / f"{ticker}{self.file_extension}"
                self._save_model(model_file, ticker)

    @typechecked
    @abstractmethod
    def _save_model(self, model_file: Path, ticker: str = None):
        """Implements saving logic for specified model type."""
        pass
        
    @typechecked
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