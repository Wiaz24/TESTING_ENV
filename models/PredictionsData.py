from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go

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
    def predictions_df(self) -> pd.DataFrame:
        """
        Zwraca dataframe z predykcjami dla wszystkich tickerów. Kolumny są tickerami, a wiersze to daty.
        """
        if not self._dataframes:
            return pd.DataFrame()
        
        pred_df = pd.DataFrame(index=self.index_df)
        for ticker, df in self._dataframes.items():
            if df.empty:
                continue
            pred_df[ticker] = df[self._predictions_col]
        pred_df = pred_df.dropna(how='all')
        pred_df.index.name = 'Date'
        return pred_df
            
    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Zwraca kopie dataframe'ów dla każdego tickera.
        """
        # Zwracamy kopie aby zapobiec modyfikacji z zewnątrz
        return {ticker: df.copy() for ticker, df in self._dataframes.items()}
    
    @property
    def check_for_nan(self) -> bool:
        """
        Sprawdza czy w dataframe'ach są wartości NaN.
        """
        for df in self._dataframes.values():
            if df.isnull().values.any():
                print(f"NaN rows in {df.index[df.isnull().any(axis=1)]}")
                return True
        return False
    
    @property
    def check_for_inf(self) -> bool:
        """
        Sprawdza czy w dataframe'ach są wartości Inf.
        """
        for df in self._dataframes.values():
            if np.isinf(df.values).any():
                return True
        return False

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

    def plot_predictions(self, ticker: str, save_path: Optional[Path] = None) -> go.Figure:
        """
        Plot the predictions and correct data for a specific ticker using plotly
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the list of tickers.")
        
        df = self._dataframes[ticker]
        if df.empty:
            raise ValueError(f"No data available for ticker {ticker}.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[self._predictions_col], mode='lines', name='Predictions'))
        fig.add_trace(go.Scatter(x=df.index, y=df[self._correct_data_col], mode='lines', name='Correct Data'))
        fig.update_layout(title=f"Predictions vs Correct Data for {ticker}",
                          xaxis_title="Date",
                          yaxis_title="Value",
                          legend_title="Legend")
        if save_path:
            fig.write_image(save_path)
        return fig
        
        