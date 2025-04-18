from abc import ABC, abstractmethod
from pathlib import Path
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
