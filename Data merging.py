import requests
import pandas as pd
from datetime import datetime

# Fetch US interest rate (FRED Funds Rate)
fedfunds_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
fedfunds_data = pd.read_csv(fedfunds_url)
fedfunds_data['DATE'] = pd.to_datetime(fedfunds_data['DATE'])
fedfunds_data.rename(columns={'FEDFUNDS': 'Interest_Rate'}, inplace=True)

# Fetch Gold Price (e.g. via FRED or other financial data providers)
gold_url = 'https://data.nasdaq.com/api/v3/datasets/LBMA/GOLD.csv' # Example link
gold_data = pd.read_csv(gold_url)
gold_data['Date'] = pd.to_datetime(gold_data['Date'])
gold_data.rename(columns={'USD (AM)': 'Gold_Price'}, inplace=True)

# Fetch Oil Price (e.g. from FRED or other sources)
oil_url = 'https://fred.stlouisfed.org/data/DCOILWTICO.csv'  # Example link
oil_data = pd.read_csv(oil_url)
oil_data['DATE'] = pd.to_datetime(oil_data['DATE'])
oil_data.rename(columns={'DCOILWTICO': 'Oil_Price'}, inplace=True)

# Fetch Fear and Greed Index (via API or other sources)
fng_url = 'https://api.alternative.me/fng/?limit=1000'
response = requests.get(fng_url)
volatility_data = pd.json_normalize(response.json(), 'data')
volatility_data['timestamp'] = pd.to_datetime(volatility_data['timestamp'], unit='s')
volatility_data.rename(columns={'value': 'Fear_Greed_Index'}, inplace=True)

# Merge datasets
btc_data = pd.read_csv('../bitcoin_weekly_2017_2024.csv')
btc_data['Timestamp'] = pd.to_datetime(btc_data['Timestamp'])

# Merge data based on dates
btc_data = pd.merge_asof(btc_data, fedfunds_data, left_on='Timestamp', right_on='DATE')
btc_data = pd.merge_asof(btc_data, gold_data[['Date', 'Gold_Price']], left_on='Timestamp', right_on='Date')
btc_data = pd.merge_asof(btc_data, oil_data[['DATE', 'Oil_Price']], left_on='Timestamp', right_on='DATE')
btc_data = pd.merge_asof(btc_data, volatility_data[['timestamp', 'Fear_Greed_Index']], left_on='Timestamp', right_on='timestamp')

# Drop extra columns
btc_data = btc_data.drop(columns=['DATE', 'Date', 'timestamp'])

# Save to CSV for modeling
btc_data.to_csv('btc_modeling_data.csv', index=False)
