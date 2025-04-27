import pandas as pd
import numpy as np
from prophet import Prophet
from models.PredictionsData import PredictionsData

ticker_df = pd.read_csv("data/tickers/EEM.csv", index_col=0, parse_dates=True)
ticker_df["next_day_close"] = ticker_df["Close"].shift(-1)
# ticker_df.info()
# ticker_df["log_return"] = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1))
# ticker_df["next_day_log_return"] = np.log(ticker_df["Close"].shift(-1) / ticker_df["Close"])
ticker_df.dropna(inplace=True)
# log_return_df.info()

train_all_df = ticker_df.iloc[:int(len(ticker_df) * 0.8)]
test_all_df = ticker_df.iloc[int(len(ticker_df) * 0.8):]

model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.add_regressor("Open")
model.add_regressor("High")
model.add_regressor("Low")
model.add_regressor("Close")
model.add_regressor("Volume")

train_df = pd.DataFrame({
    "ds": train_all_df.index,
    "y": train_all_df["next_day_close"],
    "Open": train_all_df["Open"],
    "High": train_all_df["High"],
    "Low": train_all_df["Low"],
    "Close": train_all_df["Close"],
    "Volume": train_all_df["Volume"]
})
test_df = pd.DataFrame({
    "ds": test_all_df.index,
    "Open": test_all_df["Open"],
    "High": test_all_df["High"],
    "Low": test_all_df["Low"],
    "Close": test_all_df["Close"],
    "Volume": test_all_df["Volume"]
})

print(f"Train data range: {train_df['ds'].min()} - {train_df['ds'].max()}")
print(f"Test data range: {test_df['ds'].min()} - {test_df['ds'].max()}")

model.fit(train_df)
predictions = model.predict(test_df)


# print(f"PredObject index: {predObject.minimum_date} - {predObject.maximum_date}, {len(predObject.index_df)} samples")
# print(f"Predictions index: {predictions['ds'].min()} - {predictions['ds'].max()}, {len(predictions)} samples")
# print(f"Correct data index: {test_all_df.index.min()} - {test_all_df.index.max()}, {len(test_all_df)} samples")

# if (predObject.index_df != predictions["ds"]).any():
#     raise ValueError("Predictions and PredObject index do not match.")

pred_log_return = np.log(predictions["yhat"] / test_all_df["Close"].values)
correct_log_return = np.log(test_all_df["next_day_close"].values / test_all_df["Close"].values)

pred_log_return_series = pd.Series(pred_log_return)
pred_log_return_series.index = test_df["ds"].values
pred_log_return_series.dropna(inplace=True)

correct_log_return_series = pd.Series(correct_log_return)
correct_log_return_series.index = test_df["ds"].values
correct_log_return_series.dropna(inplace=True)

predObject = PredictionsData(_index=pred_log_return_series.index, 
                             _tickers=["EEM"])

predObject.add_prediction(ticker="EEM",
                          prediction=pred_log_return_series)
predObject.add_correct_data(ticker="EEM",
                            correct_data=correct_log_return_series)

print(predObject.metrics_df)
fig = predObject.plot_predictions(ticker="EEM")
fig.show()