import sys
import os
from dataloaders.EtfDataloader import EtfDataloader

import pandas as pd
from plotly.io import show
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skfolio import Population, Portfolio
from skfolio.optimization import EqualWeighted
from skfolio.preprocessing import prices_to_returns
from skfolio.pre_selection import DropCorrelated

start_date = '2010-01-01'
split_date = '2020-01-01'
end_date = '2025-01-01'

dataloader = EtfDataloader("data/tickers", start_date, end_date)

prices = dataloader.get_close_df_for_all_tickers()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, shuffle=False, test_size=0.3)

population = Population([])
pipe = Pipeline([
    ('drop_correlated', DropCorrelated(threshold=1)),
    ('equal_weighted', EqualWeighted())
])
pipe.fit(X_train)
portfolio1 = pipe.predict(X_test)
portfolio1.name = "Equal Weighted - all assets"
population.append(portfolio1)

for threshold in [0.8, 0.85, 0.9, 0.95]:
    pipe = Pipeline([
        ('drop_correlated', DropCorrelated(threshold=threshold)),
        ('equal_weighted', EqualWeighted())
    ])
    pipe.fit(X_train)
    portfolio = pipe.predict(X_test)
    portfolio.name = f"Drop Correlated - {threshold}"

    population.append(portfolio)

fig = population.plot_cumulative_returns()
show(fig)