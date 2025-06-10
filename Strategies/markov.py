from Data.fetchdata import retrieve_stock_data, get_index_symbols
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from Config.config import config
from Utilities.garch import forecast_tgarch_risk, forecast_egarch_risk
import json
import os

# required_cols = [
#     'Company Name', 'Industry', 'Symbol', 'ISIN Code', 'prev_state', 'gap_up', 'gap_down', 'state',
#     'cum_returns', 'avg_returns', 'occurance', 'probability', 'expected_return %', "forecast_volatility", 
#     "prob_up", "prob_down", "expected_return", "max_return_3sigma", "min_return_3sigma", "ci_lower", "ci_upper", "var_95"
# ]

def get_or_generate_gap_key(symbol, last_day, interval, strong_analysis=True, json_path=config['PATHS']['TODAY_OPENING_JSON']):

    today_str = datetime.now().strftime('%Y-%m-%d')
    today_open = None

    # Step 1: Load existing JSON or initialize empty dict
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                gap_data = json.load(f)
            today_open = gap_data.get(today_str, {}).get(interval, {}).get(symbol, {}).get('Open')
            if not isinstance(gap_data, dict):
                raise ValueError("JSON content is not a dictionary")
        else:
            gap_data = {}
    except Exception as e:
        print(f"Error loading JSON: {e}")
        gap_data = {}

    # Step 2: Check if today's key already exists for the symbol
    if today_open is None:
        print('Finding Today Open Price')
        try:
            if interval == '1d':
                data = yf.download(
                    symbol,
                    start=datetime.now() - timedelta(days=1),
                    interval='1m',
                    group_by='ticker',
                    progress=False
                ).loc[today_str]
            elif interval == '1wk':
                data = yf.download(
                    symbol,
                    start=last_day['date'],
                    interval='1d',
                    group_by='ticker',
                    progress=False
                )
            print(data)

            if data.empty:
                print(f"[{symbol}] Market Not Open Yet or No Data Available")
                return None, None, None

            # Step 4: Extract today's open and compute the key
            today_open = data[symbol].iloc[0]['Open']
            # today_high = data[symbol].iloc[0]['High']
            # today_low = data[symbol].iloc[0]['Low']
            # today_close = data[symbol].iloc[0]['Close']

            print('Today\'s Open Price: ', today_open)

            # Ensure both levels exist
            # gap_data.get(today_str, {}).get(interval, {}).get(symbol, {}).get('Open')
            if today_str not in gap_data:
                gap_data[today_str] = {}

            if interval not in gap_data[today_str]:
                gap_data[today_str][interval] = {}

            if symbol not in gap_data[today_str][interval]:
                gap_data[today_str][interval][symbol] = {}

            # Now safe to assign
            gap_data[today_str][interval][symbol]['Open'] = today_open
            # gap_data[today_str][symbol][interval]['High'] = today_high
            # gap_data[today_str][symbol][interval]['Low'] = today_low
            # gap_data[today_str][symbol][interval]['Close'] = today_close

            # Write to file
            with open(json_path, 'w') as f:
                json.dump(gap_data, f, indent=2)

        except Exception as e:
            print(f"[{symbol}] Error generating key: {e}") 
            return None, None, None

    # Finding key based on parameters
    if strong_analysis:
        gap_up = today_open > last_day['high']
        gap_down = today_open < last_day['low']
    else:
        gap_up = today_open > max(last_day['open'], last_day['close'])
        gap_down = today_open < min(last_day['open'], last_day['close'])

    return last_day['state'], gap_up, gap_down

def stock_gap_analysis(symbol='INFY.NS', start_date=None, end_date=None, interval='1d', strong_analysis=True, oc_returns=True):
    """Process the OHLCV data and group it by state transitions and gaps."""
    # columns = ['Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code']

    # symbols_list = get_index_symbols(data=dataframe, read=False)
    df = retrieve_stock_data(symbol, start_date=start_date, end_date=end_date, interval=interval)
    # print('Original Dataframe:', df)

    # Drop rows with NaN values in any of the group fields
    # df = df.dropna()

    # Calculate gap up and gap down
    if strong_analysis:
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(bool)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(bool)
    else:
        df['gap_up'] = (df['open'] > np.maximum(df['open'].shift(1), df['close'].shift(1))).astype(bool)
        df['gap_down'] = (df['open'] < np.minimum(df['open'].shift(1), df['close'].shift(1))).astype(bool)
    
    # Calculate returns using open-close difference or day close percentage change
    if oc_returns:
        df['daily_returns'] = ((df['close'] - df['open']) / df['close']) * 100 
    else:
        df['daily_returns'] = df['close'].pct_change() * 100
    # print('Daily Returns:', df)

    # State definitions
    df['state'] = df['daily_returns'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
    df['prev_state'] = df['state'].shift(1)

    grouped = df.groupby(['prev_state', 'gap_up', 'gap_down', 'state']).agg(
        occurance=('daily_returns', 'count'),
        cum_returns=('daily_returns', 'sum'),
        avg_returns=('daily_returns', 'mean')
    ).reset_index()

    # print('Grouped DataFrame:', grouped)

    # Step 1: Calculate total count for each group
    grouped['key'] = grouped[['prev_state', 'gap_up', 'gap_down']].astype(str).agg('-'.join, axis=1)
    group_total = grouped.groupby('key')['occurance'].transform('sum')
    # print(group_total)

    # Step 2: Calculate probability
    grouped['probability'] = grouped['occurance'] / group_total
    # print('Grouped DataFrame with probabilities:', grouped['probability'])

    # Step 3: Calculate expected return = probability × avg_return
    # grouped['expected_return %'] = (grouped['probability'] * abs(grouped['avg_returns']))
    grouped['expected_return %'] = (grouped['probability'] * grouped['avg_returns'])
    # print('Grouped DataFrame with expected returns:', grouped['expected_return %'])

    # Optional: Drop key column if not needed
    grouped.drop(columns='key', inplace=True)

    return grouped

def get_gap_analysis_key():
    pass

def stock_gap_analysis_all(dataframe, start_date=None, end_date=None, interval='1d', strong_analysis=True, oc_returns=True, filter_by='expected_return %', state_margin=True, risk_analysis='TGARCH', read=False, store_analysis=False):
    """Perform gap analysis on each stock and return a combined DataFrame with analysis metrics and metadata."""

    results = []
    analysis_path = os.path.join(config['PATHS']['MARKOV_GAP_ANALYSIS'], datetime.now().strftime('%Y-%m-%d'), interval)
    file_ext = f"{strong_analysis} {oc_returns} {filter_by} {state_margin}.parquet"
    symbols = get_index_symbols(data=dataframe, read=read)  # Assuming this returns a list of 'Symbol's

    # while True:

    for symbol, (_, row) in zip(symbols, dataframe.iterrows()):
        print(f"\nProcessing symbol: {symbol}")
        
        try:
            df = retrieve_stock_data(symbol, start_date=start_date, end_date=end_date, interval=interval)
            print('Original Dataframe:', df)
            # df = df.dropna()

            file_path = os.path.join(analysis_path, symbol + file_ext)
            print('File Path:', file_path)
                                    
            if not os.path.exists(file_path):
                print('Path Not Exists')
                os.makedirs(analysis_path, exist_ok=True)

                if strong_analysis:
                    df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(bool)
                    df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(bool)
                else:
                    df['gap_up'] = (df['open'] > np.maximum(df['open'].shift(1), df['close'].shift(1))).astype(bool)
                    df['gap_down'] = (df['open'] < np.minimum(df['open'].shift(1), df['close'].shift(1))).astype(bool)

                if oc_returns:
                    df['daily_returns'] = ((df['close'] - df['open']) / df['close'].shift(1)) * 100
                else:
                    df['daily_returns'] = df['close'].pct_change() * 100

                # Calculate optimum margin for state classification
                returns_std = df['daily_returns'].std()
                margin = 0.2 * returns_std  # You can adjust the multiplier for optimum margin

                def classify_state(x, m=margin):
                    if x > m:
                        return 'UP'
                    elif x < -m:
                        return 'DOWN'
                    # else:
                    #     return 'NEUTRAL'

                if state_margin:
                    df['state'] = df['daily_returns'].apply(classify_state)
                else:
                    df['state'] = df['daily_returns'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
                df['prev_state'] = df['state'].shift(1)
                print('Daily Returns:', df)

                # Need to save the last row of the given df in json

                # Converting into group data
                grouped = df.groupby(['prev_state', 'gap_up', 'gap_down', 'state']).agg(
                    occurance=('daily_returns', 'count'),
                    cum_returns=('daily_returns', 'sum'),
                    avg_returns=('daily_returns', 'mean')
                ).reset_index()

                # Generating Keys
                grouped['key'] = grouped[['prev_state', 'gap_up', 'gap_down']].astype(str).agg('-'.join, axis=1)

                # Finding the group total to generate probabilities
                group_total = grouped.groupby('key')['occurance'].transform('sum')

                # Getting Probablities
                grouped['probability'] = grouped['occurance'] / group_total

                # Finding Expected Returns
                # grouped['expected_return %'] = grouped['probability'] * abs(grouped['avg_returns'])
                grouped['expected_return %'] = grouped['probability'] * grouped['avg_returns']
                # print('Grouped DataFrame with probabilities:', grouped)
                
                # Saving the analysis to folder
                grouped.to_parquet(os.path.join(analysis_path, symbol + file_ext), index=False)

            else:
                grouped = pd.read_parquet(os.path.join(analysis_path, symbol + file_ext))

                if strong_analysis:
                    df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(bool)
                    df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(bool)
                else:
                    df['gap_up'] = (df['open'] > np.maximum(df['open'].shift(1), df['close'].shift(1))).astype(bool)
                    df['gap_down'] = (df['open'] < np.minimum(df['open'].shift(1), df['close'].shift(1))).astype(bool)

                if oc_returns:
                    df['daily_returns'] = ((df['close'] - df['open']) / df['close']) * 100
                else:
                    df['daily_returns'] = df['close'].pct_change() * 100

                df['state'] = df['daily_returns'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
                df['prev_state'] = df['state'].shift(1)
                
            if not store_analysis:
                print('Grouped DataFrame:', grouped)
                print('Last Row:', df.iloc[-1])

                state, gap_up, gap_down = get_or_generate_gap_key(symbol, df.iloc[-1], interval=interval, strong_analysis=strong_analysis)
                key = f"{state}-{gap_up}-{gap_down}"
                if not key:
                    print(f"No Key Generated for {symbol}")
                    continue
                print(f"[{symbol}] Generated Key: {key}")


                # Filter by current key
                analysis = grouped[grouped['key'] == key].copy()

                if analysis.empty:
                    print(f"No Gap Analysis Found For {symbol}")
                    continue
                
                # Take only the row with maximum probability for the current key
                analysis = analysis.loc[analysis[filter_by].idxmax()].to_frame().T

                # Adding Risk Analysis Algorithm
                garch_returns = df[
                    (df['prev_state'] == state) &
                    (df['gap_up'] == gap_up) &
                    (df['gap_down'] == gap_down)
                ]['daily_returns'].dropna()
                
                if garch_returns.ndim != 1:
                    raise ValueError("Input to arch_model must be 1-dimensional")
                
                if risk_analysis == 'TGARCH':
                    garch_results = forecast_tgarch_risk(garch_returns)
                elif risk_analysis == 'EGARCH':
                    garch_results = forecast_egarch_risk(garch_returns)

                # Add metadata
                for col in ['Company Name', 'Industry', 'Symbol', 'ISIN Code']:
                    analysis[col] = row.get(col, np.nan)

                for k, v in garch_results.items():
                    analysis[k] = v

                results.append(analysis)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if results and not store_analysis:
        dataframe = pd.concat(results, ignore_index=True)
        return dataframe
        # else:
        #     continue  # Return empty DataFrame if no results



# Incomplete Code
# def transition_prob_matrix(data, gap):
#     """
#     Perform gap analysis on the given data.

#     Args:
#         data (list): The data to analyze.
#         gap (int): The gap size.

#     Returns:
#         list: The result of the gap analysis.
#     """
#     result = []
#     for i in range(len(data) - gap):
#         result.append(data[i] + data[i + gap])
#     return result