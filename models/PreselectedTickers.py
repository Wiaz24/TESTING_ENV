from .PredictionsData import PredictionsData
from dataclasses import dataclass, field
import pandas as pd
import plotly.graph_objects as go

@dataclass
class PreselectedTickers:
    predictions: PredictionsData = field(default=None, repr=False, init=True)
    cardinality: int = field(default=7, repr=False, init=True)

    _selected_tickers: pd.Series = field(default_factory=pd.Series, repr=False, init=False)

    @property
    def minimum_date(self) -> pd.Timestamp:
        """Zwraca minimalną datę."""
        return self._selected_tickers.index.min()
    
    @property
    def maximum_date(self) -> pd.Timestamp:
        """Zwraca maksymalną datę."""
        return self._selected_tickers.index.max()

    def __post_init__(self):
        if self.cardinality <= 0:
            raise ValueError("Cardinality must be a positive integer.")

        # Initialize the selected tickers
        self._selected_tickers = pd.Series(index=self.predictions.index_df, dtype=object)
        self._select_tickers()

    def _select_tickers(self):
        """
        Select the top N tickers based on the predictions.
        """
        predictions_df = pd.DataFrame(index=self.predictions.index_df)
        for ticker, df in self.predictions.dataframes.items():
            predictions_df[ticker] = df[self.predictions._predictions_col]

        # Sort the predictions and select the top N tickers
        for date, row in predictions_df.iterrows():
            sorted_tickers = row.nlargest(self.cardinality).index
            self._selected_tickers[date] = sorted_tickers

    def filter_close_df(self, df: pd.DataFrame, test_date: pd.Timestamp) -> pd.DataFrame:
        """
        Return a dataframe only with the selected tickers for the given date.
        """
        filtered_df = pd.DataFrame(index=df.index)
        if test_date not in self._selected_tickers.index:
            raise ValueError(f"Test date {test_date} not found in selected tickers index.")
        selected_tickers_for_date = self._selected_tickers[test_date]
        for ticker in selected_tickers_for_date:
            if ticker in df.columns:
                filtered_df[ticker] = df[ticker]
            else:
                print(f"Warning: Ticker {ticker} not found in the provided dataframe.")
        filtered_df.index.name = "Date"
        return filtered_df
    
    def plot_selection_histogram(self) -> go.Figure:
        """
        Tworzy histogram pokazujący częstość wyboru poszczególnych tickerów.
        
        Returns:
            go.Figure: Wykres Plotly przedstawiający histogram wyboru tickerów.
        """
        all_tickers = self.predictions.tickers
        
        # Tworzenie słownika zliczającego wystąpienia każdego tickera
        ticker_counts = {ticker: 0 for ticker in all_tickers}
        
        # Zliczanie wystąpień tickerów w _selected_tickers
        for selected_list in self._selected_tickers:
            for ticker in selected_list:
                if ticker in ticker_counts:
                    ticker_counts[ticker] += 1
                else:
                    print(f"Warning: Ticker {ticker} not found in the all_tickers list.")
        
        # Tworzenie wykresu histogramu
        fig = go.Figure(data=[go.Bar(
            x=list(ticker_counts.keys()),
            y=list(ticker_counts.values()),
            marker=dict(color='blue')
        )])
        fig.update_layout(
            title="Histogram wyboru tickerów",
            xaxis_title="Tickery",
            yaxis_title="Liczba wystąpień",
            xaxis_tickangle=-45,
            showlegend=False
        )

        return fig
       