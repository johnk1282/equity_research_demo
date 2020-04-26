# equity_research_demo
Tools and analytics to construct stock-level features, backtest signals, and optimize portfolios of signals, using common academic data sources. Jupyter Notebook (demo.ipynb) demonstrates some of the functionality of these research tools.

Files:

1. pull_data.py: functions to import raw data from WRDS API; datasets include CRSP for pricing, Compustat for fundamental data, IBES for analyst estimates, Thomson Reuters for 13F data
2. create_feature_df.py: functions to calculate stock characteristics (features) from input data, lag appropriately, and merge onto standardized dataframe; mostly replicating well-known academic papers + variations
3. test_signals.py: Backtester class with functions to create signals from features (rank features within a particular universe), perform analytics based on backtests, and output finalized models (collections of signals)
4. optimize_portfolios.py: SecurityRanker class with functions to combine signals based on different optimization frameworks; PortfolioConstructor class that gets final portfolio weights by imposing basic portfolio construction constraints on the security ranking
