import ccxt
from datetime import datetime
import pandas as pd
import requests
import yfinance as yf

# Binance API-Schlüssel und API-Geheimnis einfügen
api_key = 'G3FE5OP45RXiQ21ma6ZOC2xcchF3iPeVzz7pPxA7opWd1xtSGrPFZSI7l4R70rQa'
api_secret = 'ymP7iZHFN9Q8Zb7mM6OANh3PPEgQzeWJkltMGGzzk61shfYFsW7pwgEUzSzWu5oX'

# Binance Instanz erstellen
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True  # Stellt sicher, dass API-Anfragen nicht die Ratenbegrenzung überschreiten
})

# Funktion zur Abfrage der OHLCV-Daten in Intervallen
def get_ohlcv_interval(symbol='BTC/USDT', timeframe='1w', start_date=None, end_date=None):
    all_data = []
    since = int(start_date.timestamp() * 1000)  # Konvertiere Startdatum in Millisekunden-Timestamp
    end_timestamp = int(end_date.timestamp() * 1000)  # Enddatum in Millisekunden-Timestamp

    while since < end_timestamp:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_data += ohlcv
            since = ohlcv[-1][0] + 60 * 60 * 24 * 7  # Setze das "since"-Datum auf das letzte abgerufene Datum + 1 Woche
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            break

    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')  # Konvertiere die Zeit in datetime-Format
    df = df[df['Timestamp'] <= end_date]

    return df

# Fetching Bitcoin price and volume (2017-2024)
start_date = datetime(2017, 1, 1)
end_date = datetime(2024, 9, 30)
btc_weekly_data = get_ohlcv_interval(symbol='BTC/USDT', start_date=start_date, end_date=end_date)

# Fetch US interest rate (FRED Funds Rate) - with filtering for necessary dates
fedfunds_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
try:
    fedfunds_data = pd.read_csv(fedfunds_url)
    fedfunds_data['DATE'] = pd.to_datetime(fedfunds_data['DATE'])
    fedfunds_data = fedfunds_data[(fedfunds_data['DATE'] >= '2017-01-01') & (fedfunds_data['DATE'] <= '2024-09-30')]
    fedfunds_data.rename(columns={'FEDFUNDS': 'Interest_Rate'}, inplace=True)
except Exception as e:
    print(f"Error fetching Fed Funds Rate data: {e}")
    fedfunds_data = pd.DataFrame()

# Fetch Bitcoin Fear and Greed Index (Volatility)
volatility_url = 'https://api.alternative.me/fng/?limit=1000'
try:
    response = requests.get(volatility_url)
    if response.status_code == 200:
        volatility_data = pd.json_normalize(response.json(), 'data')
        volatility_data['timestamp'] = pd.to_numeric(volatility_data['timestamp'], errors='coerce')
        volatility_data['timestamp'] = pd.to_datetime(volatility_data['timestamp'], unit='s')
        volatility_data = volatility_data.sort_values(by='timestamp')  # Sort the data
        volatility_data.rename(columns={'value': 'Volatility'}, inplace=True)
        volatility_data['Volatility'] = pd.to_numeric(volatility_data['Volatility'])
    else:
        print(f"Failed to fetch Fear and Greed Index, status code: {response.status_code}")
        volatility_data = pd.DataFrame()
except Exception as e:
    print(f"Error fetching Fear and Greed Index: {e}")
    volatility_data = pd.DataFrame()

# Synchronizing and merging the data
btc_data = pd.merge_asof(btc_weekly_data.sort_values('Timestamp'),
                         fedfunds_data.sort_values('DATE'),
                         left_on='Timestamp', right_on='DATE')

btc_data = pd.merge_asof(btc_data,
                         volatility_data[['timestamp', 'Volatility']].sort_values('timestamp'),
                         left_on='Timestamp', right_on='timestamp')

btc_data = btc_data.drop(columns=['DATE', 'timestamp'])

# Handle missing values (fill forward or drop rows)
btc_data.ffill(inplace=True)  # Verwende ffill ohne fillna

# Save the data to CSV for the modeling part
btc_data.to_csv('btc_modeling_data.csv', index=False)

print(btc_data.head())  # Check the final dataset
