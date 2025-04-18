import pandas as pd
from prediction_models.IPredictionModel import IPredictionModel
from models.Predictions import Predictions

class AssetSelector():
    def __init__(self, tickers: list[str], cardinality: int):
        self.tickers = tickers
        self.cardinality = cardinality

    def select_assets(self, predictions: Predictions) -> list[str]:
        """
        Selects the top N assets based on the model's predictions.
        """
        for ticker, preds in predictions.predictions.items():
            if len(preds) != 1:
                raise ValueError(f"Predictions for ticker {ticker} should be a single value.")
        # Sort the predictions in descending order
        sorted_predictions = sorted(predictions.predictions.items(), key=lambda x: x[1], reverse=True)
        # Select the top N assets
        selected_assets = [ticker for ticker, _ in sorted_predictions[:self.cardinality]]
        return selected_assets
    
    def filter_df_for_selected_assets(self, df: pd.DataFrame, selected_assets: list[str]) -> pd.DataFrame:
        """
        Filter the dataframe for the selected assets.
        """
        if not all(asset in df.columns for asset in selected_assets):
            raise ValueError("Some selected assets are not present in the dataframe.")
        return df[selected_assets]