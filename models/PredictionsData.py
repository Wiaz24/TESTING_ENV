from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

@dataclass
class PredictionsData:
    _predictions_col: str = field(default='Predictions', repr=False, init=False)
    _correct_data_col: str = field(default='CorrectData', repr=False, init=False)
    _index: pd.Index = field(default=None, repr=True, init=True)
    _tickers: List[str] = field(default_factory=list, repr=True, init=True)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        for ticker in self._tickers:
            if ticker not in self._dataframes:
                self._dataframes[ticker] = pd.DataFrame(index=self._index)
                self._dataframes[ticker][self._predictions_col] = pd.Series(dtype=np.float64)
                self._dataframes[ticker][self._correct_data_col] = pd.Series(dtype=np.float64)

    @property
    def tickers(self) -> List[str]:
        """Zwraca listę tickerów."""
        return self._tickers.copy()
    
    @property
    def minimum_date(self) -> pd.Timestamp:
        """Zwraca minimalną datę."""
        min_date = pd.Timestamp.min

        for df in self._dataframes.values():
            if not df.empty:
                min_date = max(min_date, df.index.min())
        return min_date
    
    @property
    def maximum_date(self) -> pd.Timestamp:
        """Zwraca maksymalną datę."""
        max_date = pd.Timestamp.max

        for df in self._dataframes.values():
            if not df.empty:
                max_date = min(max_date, df.index.max())
        return max_date
    
    @property
    def metrics_df(self) -> pd.DataFrame:
        mdf = pd.DataFrame(  index=self._dataframes.keys(), 
                            columns=['MSE', 'RMSE', 'MAE', 'MAPE'], 
                            dtype=float)
        mdf = mdf.fillna(0)
        mdf.index.name = 'Ticker'

        for ticker, df in self._dataframes.items():
            if df.empty:
                continue
            y_test = df[self._correct_data_col]
            y_pred = df[self._predictions_col]
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Pomnóż przez 100 dla procentów

            mdf.loc[ticker] = [mse, rmse, mae, mape]
        return mdf

    @property
    def index_df(self) -> pd.Index:
        """
        Zwraca indeks dataframe'ów.
        """
        if self._dataframes:
            return next(iter(self._dataframes.values())).index
        return pd.Index([])
    
    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Zwraca kopie dataframe'ów dla każdego tickera.
        """
        # Zwracamy kopie aby zapobiec modyfikacji z zewnątrz
        return {ticker: df.copy() for ticker, df in self._dataframes.items()}
    def add_prediction(self, ticker: str, prediction: pd.Series):
        """
        Add a prediction for a specific ticker.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the list of tickers.")
        if len(prediction) == 0:
            raise ValueError(f"Prediction for ticker {ticker} is empty.")
        if prediction.index is not self._dataframes[ticker].index:
            raise ValueError(f"Prediction index for ticker {ticker} does not match the dataframe index.")
        
        self._dataframes[ticker][self._predictions_col] = prediction

    def add_correct_data(self, ticker: str, correct_data: pd.Series):
        """
        Add correct data for a specific ticker.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the list of tickers.")
        if len(correct_data) == 0:
            raise ValueError(f"Correct data for ticker {ticker} is empty.")
        if correct_data.index is not self._dataframes[ticker].index:
            raise ValueError(f"Correct data index for ticker {ticker} does not match the dataframe index.")
        
        self._dataframes[ticker][self._correct_data_col] = correct_data