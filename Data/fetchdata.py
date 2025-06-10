import sqlite3
from Config.config import config
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd


def fetch_and_store_multiple_stocks(ticker_list, start_date=None, end_date=None, db_path = config['PATHS']['STOCK_DATABASE'], interval='1d'):
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {config['DATABASE'][interval]} (
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()

    # Download all tickers at once
    try:
        if not (start_date or end_date):
            stock_data = yf.download(ticker_list, period='max', group_by='ticker', auto_adjust=False, threads=True, interval=interval, end=datetime.now() - timedelta(days=0))
        else: 
            stock_data = yf.download(ticker_list, start=start_date, end=end_date, group_by='ticker', auto_adjust=False, threads=True, interval=interval)
    except Exception as e:
        print(f"Failed to download stock data: {e}")
        conn.close()
        return

    if stock_data.empty:
        print("No data downloaded.")
        conn.close()
        return

    # Handle both single and multiple tickers correctly
    if len(ticker_list) == 1:
        # stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Single ticker case
        ticker = ticker_list[0]
        rows_to_insert = []
        stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close'])  # Drop rows with NaN values
        for index, row in stock_data.iterrows():
            rows_to_insert.append((
                ticker,
                index.strftime("%Y-%m-%d"),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))
        cursor.executemany(f"""
            INSERT OR IGNORE INTO {config['DATABASE'][interval]} (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        print(f"Inserted {cursor.rowcount} rows for {ticker} into the database.")
    else:
        # Multiple tickers case
        for ticker in ticker_list:
            if ticker not in stock_data.columns.get_level_values(0).unique():
                print(f"No data for {ticker}. Skipping.")
                continue
            rows_to_insert = []
            ticker_data = stock_data[ticker]
            ticker_data = ticker_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            # ticker_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for index, row in ticker_data.iterrows():
                rows_to_insert.append((
                    ticker,
                    index.strftime("%Y-%m-%d"),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            cursor.executemany(f"""
                INSERT OR IGNORE INTO {config['DATABASE'][interval]} (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows_to_insert)
            print(f"Inserted {cursor.rowcount} rows for {ticker} into the database.")

    conn.commit()
    conn.close()
    print("✅ All tickers processed successfully.")

def retrieve_stock_data(ticker_list, start_date=datetime.now().date() - timedelta(days=365), end_date=datetime.now().date(), db_path = config['PATHS']['STOCK_DATABASE'], interval='1d'):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Handle single ticker as list
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]

    placeholders = ','.join(['?'] * len(ticker_list))  # Creates (?, ?, ?) dynamically based on number of tickers

    query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM {config['DATABASE'][interval]}
        WHERE ticker IN ({placeholders})
    """

    params = list(ticker_list)

    # Add optional date filtering
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY ticker, date ASC"

    # Execute the query
    try:
        stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        
    if stock_data.empty:
        print("No data found in database. Downloading from Yahoo Finance...")
        fetch_and_store_multiple_stocks(ticker_list, interval=interval)
        stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])       
    
    # print(stock_data.head())

    conn.close()

    return stock_data

def get_index_symbols(csv_path=r"Data\INDEXES\Nifty Total Market.csv", read=True, data=None, symbol_column='Symbol'):
    if read:
        data = pd.read_csv(csv_path)
    if symbol_column not in data.columns:
        raise ValueError("CSV must contain a 'Symbol' column.")
    symbols = [f"{sym.strip()}.NS" for sym in data[symbol_column]]
    return symbols

def fetch_data_for_symbols_df(symbols_df, symbol_column='Symbol', start_date=None, end_date=None, interval='1d'):
    """
    Fetch and store stock data for all symbols in a DataFrame using fetch_and_store_multiple_stocks.
    Args:
        symbols_df (pd.DataFrame): DataFrame containing stock symbols.
        symbol_column (str): Name of the column containing symbols.
        start_date (str): Start date for fetching data.
        end_date (str): End date for fetching data.
        interval (str): Data interval (e.g., '1d').
    """ 
    # Ticker List Example: ['INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
    ticker_list = get_index_symbols(data=symbols_df, read=False, symbol_column=symbol_column)
    print(f"Ticker List: {ticker_list}")
    fetch_and_store_multiple_stocks(ticker_list, start_date=start_date, end_date=end_date, interval=interval)

# if __name__ == "__main__":
#     # Example usage
#     ticker_list = get_index_symbols()
#     start_date = "1980-01-01"
#     end_date = "2025-05-01"
#     # end_date = datetime.now().strftime('%Y-%m-%d') 

#     fetch_and_store_multiple_stocks(ticker_list, start_date, end_date)
#     data = retrieve_stock_data(ticker_list, start_date, end_date)
#     data.head()