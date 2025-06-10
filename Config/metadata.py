import yfinance as yf
import json
import os
import sqlite3
from datetime import datetime

# List of stock symbols to process
stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Path to output JSON file
metadata_file_path = "./Config/stock_metadata.json"
log_file_path = "./Config/failed_stocks.log"

def get_stock_metadata(stock_symbol, interval="1d"):
    """
    Download historical data for the stock and extract metadata.
    """
    try:
        data = yf.download(stock_symbol, interval=interval, progress=False)
        if data.empty:
            raise ValueError("No data available")

        metadata = {
            "stock_symbol": stock_symbol,
            "start_date": str(data.index[0].date()),
            "end_date": str(data.index[-1].date()),
            "total_records": len(data),
            "interval": interval,
            "last_updated": datetime.now().isoformat(timespec='seconds')
        }
        return metadata

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        log_failed_stock(stock_symbol)
        return None

def load_existing_metadata(file_path):
    """
    Load existing JSON metadata if it exists, or return empty dict.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def log_failed_stock(stock_symbol):
    """
    Log failed stock symbol to a separate file with timestamp.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now().isoformat()} - Failed to fetch: {stock_symbol}\n")

def update_metadata_file(stock_list, file_path, interval="1d"):
    """
    Update the JSON metadata file with new or updated stock entries.
    """
    all_metadata = load_existing_metadata(file_path)

    for symbol in stock_list:
        metadata = get_stock_metadata(symbol, interval)
        if metadata:
            key = f"{symbol}_{interval}"
            all_metadata[key] = metadata

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Metadata updated and saved to: {file_path}")

# Run the full update process
if __name__ == "__main__":
    update_metadata_file(stock_list, metadata_file_path)
