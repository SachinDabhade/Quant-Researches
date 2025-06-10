import streamlit as st
from Data.fetchdata import retrieve_stock_data, fetch_and_store_multiple_stocks, fetch_data_for_symbols_df, get_index_symbols
import pandas as pd
from streamlit_option_menu import option_menu
from Config.config import config
from Config.fetchindex import fetch_all_nse_index_csv, fetch_nse_index_csv
from Config.indexjson import csv_to_json, get_index_json, get_index_url
import plotly.graph_objects as go
from Strategies.markov import stock_gap_analysis, stock_gap_analysis_all
import datetime
import os

required_cols = [
    'Company Name', 'Symbol', "prob_up", "prob_down", "forecast_volatility", "expected_return", "ci_lower", "ci_upper", "var_95", 'avg_returns', 'occurance', 'expected_return %', 'probability', 'state'
]

st.set_page_config(page_title="Quant Researches Dashboard", layout="wide")
with st.sidebar:
    option = option_menu(
        "Navigation",
        ["Dashboard", "Stock Scanner", "Strategy Analysis", "Market Data", "Portfolio", "Settings"],
        icons=["speedometer", "search", "bar-chart", "graph-up", "briefcase", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px"},
            "icon": {"font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
            },
            "nav-link-selected": {"background-color": "#636efa", "color": "white"},
        },
    )

st.title("Quant Researches - A Pilot Project")

if option == "Dashboard":
    # from streamlit_option_menu import option_menu
    st.header("Dashboard Overview")
    st.write("Welcome to your trading dashboard. Here you can monitor key metrics and performance.")
    st.subheader("Stock Search & Candlestick Visualization")

    with st.container():
        col_stock, col_start, col_end, col_interval, col_filter = st.columns([1.2, 1, 1, 1, 1.8])
        with col_stock:
            stock_symbol = st.selectbox(                
                "Stock",
                get_index_symbols(os.path.join(config['PATHS']['INDEXES_DIR'], "Nifty Total Market.csv"), read=True, data=None, symbol_column='Symbol'),
                key="stock_input",
                index=20,
                # label_visibility="collapsed",
                help="Enter NSE Stock Symbol",
            )
        with col_start:
            start_date = st.date_input(
                "Start",
                value=datetime.date.today() - datetime.timedelta(days=365),
                min_value=datetime.date(1980, 1, 1),
                max_value=datetime.date.today(),
                key="start_date_compact",
                # label_visibility="collapsed"
            )
        with col_end:
            end_date = st.date_input(
                "End",
                value=datetime.date.today(),
                min_value=start_date,
                max_value=datetime.date.today(),
                key="end_date_compact",
                # label_visibility="collapsed"
            )
        with col_interval:
            interval = st.selectbox(
            "Interval",
            config['VARIABLES']['INTERVALS'].split(","),
            index=8,  # default to "1d"
            key="interval_select",
            help="Select data interval"
            )
        with col_filter:
            filter_options = ["Markov Chain", "Candlestick Patterns", "Trend Analysis", "Volume"]
            selected_filters = st.multiselect(
                "Filters",
                filter_options,
                default=["Markov Chain"],
                key="visualization_filters_compact",
                help="Select filters",
                placeholder="Choose filters",
                # label_visibility="collapsed"
            )

    # No visualization logic here
    if stock_symbol:
        # stock_symbol = stock_symbol.upper() + '.NS'
        try:
            if start_date and end_date:
                start_date = pd.to_datetime(start_date).date()
                end_date = pd.to_datetime(end_date).date()
                symbol_data = retrieve_stock_data(stock_symbol, start_date=start_date, end_date=end_date, interval=interval)
                print('Stock Data :', symbol_data)
            if not symbol_data.empty:
                # Use the correct column name for date, or use the index if it's a DatetimeIndex
                fig = go.Figure()
                fig.update_layout(xaxis_rangeslider_visible=True, xaxis_autorange=True, yaxis_autorange=True)
                fig = go.Figure(data=[go.Candlestick(
                    x=symbol_data['date'],
                    open=symbol_data['open'],
                    high=symbol_data['high'],
                    low=symbol_data['low'],
                    close=symbol_data['close']
                )])
                fig.update_layout(height=550)
                fig.update_layout(title=f'Candlestick Chart for {stock_symbol}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for the entered symbol.")
        except Exception as e:
            st.error(f"Error fetching or plotting data: {e}")

    if "Markov Chain" in selected_filters:
        col_oc, col_strong, a, b, c = st.columns([1.2, 1, 1, 1, 1])
        with col_oc:
            oc_returns = st.toggle("Open-Close Returns", value=True, help="Show open-close returns")
        with col_strong:
            strong_analysis = st.toggle("Strong Analysis", value=True, help="Enable strong analysis")
        st.markdown(f"### Markov Chain Gap Analysis for {stock_symbol}")
        try:
            results = stock_gap_analysis(stock_symbol, start_date=None, end_date=None, interval='1d', strong_analysis=strong_analysis, oc_returns=oc_returns)
            st.dataframe(results, use_container_width=True)
        except Exception as E:
            st.error(f"Something Wents Wrong: {E}")

elif option == "Stock Scanner":
    st.header("Stock Scanner")
    # Arrange all filter inputs in a single row
    col_index, col1, col2, col3 = st.columns([1.2, 1.2, 1.2, 1])
    with col_index:
        # index_options = pd.read_csv(config['PATHS']['INDEX_LINKS'])["Indexes"].tolist()
        index_options = get_index_json().keys()
        selected_index = st.selectbox(
            "Index",
            index_options,
            index=4,
            key="scanner_index_select",
            help="Select the stock index"
        )
        if selected_index:
            try:
                dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
            except Exception as E:
                print('CSV FILE NOT FOUND, DOWNLOADING CSV FILE FROM SERVER...!')
                fetch_nse_index_csv(get_index_url(selected_index)['URL'], selected_index)
                dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
    with col1:
        filter_options = [
            "Markov Chain",
            "Candlestick Patterns",
            "Trend Analysis",
            "Price Action",
            "Volume Analysis"
        ]
        selected_filter = st.selectbox(
            "Analysis Type",
            filter_options,
            key="scanner_filter_select",
            help="Choose the type of analysis"
        )
    with col2:
        markov_options = ["Gap Analysis", "State Transition", "Custom"]
        candle_patterns = ["Doji", "Hammer", "Engulfing", "Morning Star", "All Patterns"]
        trend_options = ["Moving Average", "Breakout", "Reversal", "Momentum"]
        price_action_options = ["Support/Resistance", "Breakout", "Pullback"]
        volume_options = ["Volume Spike", "Volume Dry Up", "OBV"]

        # Show the relevant input based on selected_filter
        if selected_filter == "Markov Chain":
            selected_filter = st.selectbox(
                "Markov Chain",
                markov_options,
                key="markov_chain_option_select",
                help="Select a Markov Chain analysis"
            )
        elif selected_filter == "Candlestick Patterns":
            selected_option = st.multiselect(
                "Patterns",
                candle_patterns,
                default=[],
                key="candlestick_patterns_select",
                help="Select candlestick patterns"
            )
        elif selected_filter == "Trend Analysis":
            selected_option = st.selectbox(
                "Trend Type",
                trend_options,
                key="trend_analysis_select",
                help="Select trend analysis type"
            )
        elif selected_filter == "Price Action":
            selected_option = st.selectbox(
                "Price Action",
                price_action_options,
                key="price_action_select",
                help="Select price action pattern"
            )
        elif selected_filter == "Volume Analysis":
            selected_option = st.selectbox(
                "Volume Type",
                volume_options,
                key="volume_analysis_select",
                help="Select volume analysis type"
            )
        else:
            selected_option = None
    
    with col3:
        interval_options = config['VARIABLES']['INTERVALS'].split(",")
        selected_interval = st.selectbox(
            "Interval",
            interval_options,
            index=8 if len(interval_options) > 8 else 0,
            key="scanner_interval_select",
            help="Select data interval"
        )
    
    if "Industry" in dataframe.columns:
        industries = dataframe["Industry"].dropna().unique().tolist()
        selected_industries = st.multiselect(
        "Industry",
        industries,
        default=[],
        key="industry_multiselect"
        )
        if selected_industries:
            dataframe = dataframe[dataframe["Industry"].isin(selected_industries)]

    # Add important buttons for analysis/filter
    col_sort, mult_sort, garch = st.columns([1, 2, 1])
    with col_sort:
        sort_by = st.selectbox(
            "Result Analyzed Using",
            ["Probability", "Avg Return", "Expected Return"],
            index=2,
            key="sort_by_select",
            help="Sort results by selected metric"
        )
        # Ensure columns exist before sorting
        sort_map = {
            "Probability": "probability",
            "Avg Return": "avg_returns",
            "Expected Return": "expected_return %"
        }
        sort_col = sort_map.get(sort_by, "expected_return %")
    with mult_sort:
        # Allow user to select multiple sort columns
        multi_sort_cols = st.multiselect(
            "Multi-Columnar Sorting",
            options=required_cols,
            # default=['state'] if sort_col in required_cols else [],
            default=[sort_col],
            key="multi_sort_by_select",
            help="Select one or more columns to sort by"
        )
    with garch:
        risk_analysis = st.selectbox(
            "RISK ANALYSIS",
            options=['TGARCH', 'EGARCH'],
            index=1,
            key="risk_analysis_algorithm",
            help="Help you to analyze the risk behind each trade"
        )
    
    col_oc, col_strong, col_fno, st_margin, x = st.columns([1.2, 1, 1, 1, 1])
    with col_oc:
        oc_returns = st.toggle("Open-Close Returns", value=True, help="Show open-close returns")
    with col_strong:
        strong_analysis = st.toggle("Strong Analysis", value=True, help="Enable strong analysis")
    with col_fno:
        fno_only = st.toggle("F&O Stocks Only", value=False, help="Show only F&O stocks")
    with st_margin:
        state_margin = st.toggle("State Martin", value=False, help="Enable/disable State Martin analysis")

    
    # if fno_only and "FNO" in dataframe.columns:
    #     dataframe = dataframe[dataframe["FNO"] == True]
    # Only run analysis if button is pressed
    if selected_index and selected_filter == 'Gap Analysis':
        results = stock_gap_analysis_all(
            dataframe,
            start_date=None,
            end_date=None,
            interval=selected_interval,
            strong_analysis=strong_analysis,
            oc_returns=oc_returns,
            filter_by=sort_col,
            state_margin=state_margin,
            risk_analysis=risk_analysis
        )
        if not isinstance(results, pd.DataFrame):
            st.error('ANALYSIS DID NOT RETURNED THE DATAFRAME >> PLEASE CHECK THE CODE BEFORE PROCEED...!')
        elif results.empty:
            st.error('NO ANALYSIS FOUND >> PLEASE CHECK THE CODE AND CHECK REQUIREMENTS IF NOT DONE PROPERLY...!')
        elif sort_col in results.columns:
            # Sort results based on selected column
            if multi_sort_cols:
                results = results.sort_values(by=multi_sort_cols, ascending=[False]*len(multi_sort_cols))
            else:
                results = results.sort_values(by=sort_col, ascending=False)
                print(results)
            results = results[results['occurance'] > 30]
            st.dataframe(results[required_cols], use_container_width=True)
        else:
            st.error('SORT COLUMNS NOT INTO RESULT TABLE...!')

elif option == "Strategy Analysis":
    st.header("Strategy Analysis")
    st.write("Analyze and backtest your trading strategies here.")
    st.line_chart([10, 20, 15, 30, 25])

elif option == "Market Data":
    st.subheader("Market Data")
    st.write("Here you can view and analyze market data.")
    
    col1, col2, col3, col4 = st.columns([1.2, 0.8, 1.5, 0.5])
    with col1:
        selected_index = st.selectbox(
            "Choose Index",
            get_index_json().keys(),
            index=104,
            help="Select a stock market index"
        )
        if selected_index:
            try:
                dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
            except Exception as E:
                print('CSV FILE NOT FOUND, DOWNLOADING CSV FILE FROM SERVER...!')
                fetch_nse_index_csv(get_index_url(selected_index)['URL'], selected_index)
                dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
    with col2:
        selected_timeframe = st.selectbox(
            "Interval",
            config['VARIABLES']['INTERVALS'].split(","),
            index=8,
            help="Select data timeframe"
        )
    with col3:
        selected_data_type = st.selectbox(
            "Searching Purpose",
            ["Historical Data", "Fundamental Data", "Update Analysis"],
            help="Select data purpose for updation"
        )
    with col4:
        st.write("")  # For alignment
        update_market_data = st.button(
            "Update Data",
            key="market_data_update_btn",
            help="Click to refresh market data",
        )
    if update_market_data:
        if selected_data_type == "Historical Data":
            try:
                dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
                fetch_data_for_symbols_df(dataframe, interval=selected_timeframe)
                st.success("Market Data Updated Successfully...!")
            except Exception as e:
                st.error(f"Failed to update market data: {e}")
        elif selected_data_type == "Fundamental Data":
            st.success('Fundamental Data Updated Successfully...!')
        elif selected_data_type == "Update Analysis":
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='expected_return %', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='expected_return %', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=False, filter_by='expected_return %', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=False, filter_by='expected_return %', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='expected_return %', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='expected_return %', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='expected_return %', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='expected_return %', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='probability', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='probability', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=False, filter_by='probability', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=False, filter_by='probability', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='probability', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='probability', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='probability', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='probability', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='avg_returns', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='avg_returns', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=False, filter_by='avg_returns', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=False, filter_by='avg_returns', state_margin=True, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='avg_returns', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='avg_returns', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=True, oc_returns=True, filter_by='avg_returns', state_margin=False, store_analysis=True)
            stock_gap_analysis_all(dataframe=dataframe, interval=selected_timeframe, strong_analysis=False, oc_returns=True, filter_by='avg_returns', state_margin=False, store_analysis=True)
            st.success('Analysis Updated Successfully...!')

    st.write("")

    col5, col6, col7 = st.columns([1, 1, 1])
    
    with col5:
        st.write("Update Indexes")  # Placeholder for future widgets or alignment
        if st.button("Update Index CSV", key="update_index_csv_btn", help="Update index csv file from web"):
            try:
                fetch_all_nse_index_csv()
                st.success("Index CSV file updated successfully!")
            except Exception as e:
                st.error(f"Failed to update JSON: {e}")
    
    with col6:
        st.write("Update JSON")  # For alignment
        if st.button("Update Index JSON", key="update_index_json_btn", help="Update index JSON file from CSV"):
            try:
                csv_path = config['PATHS']['INDEX_LINKS']
                json_path = config['PATHS']['INDEX_JSON']
                csv_to_json(csv_path, json_path)
                st.success("Index JSON file updated successfully!")
            except Exception as e:
                st.error(f"Failed to update JSON: {e}")

    with col7:
        st.write("Update ")  # For alignment
        if st.button("Update Index METADATA", key="update_index_metadata_btn", help="Update index metadata file from database"):
            st.success("Metadata Updated Successfully")
            # try:
            #     csv_path = config['PATHS']['INDEX_LINKS']
            #     json_path = config['PATHS']['INDEX_JSON']
            #     csv_to_json(csv_path, json_path)
            #     st.success("Index JSON file updated successfully!")
            # except Exception as e:
            #     st.error(f"Failed to update JSON: {e}")

    if selected_index:
        try:
            dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
        except Exception as e:
            print('CSV FILE NOT FOUND, DOWNLOADING CSV FILE FROM SERVER...!')
            fetch_nse_index_csv(get_index_url(selected_index)['URL'], selected_index)
            dataframe = pd.read_csv(os.path.join(config['PATHS']['INDEXES_DIR'], selected_index + ".csv"))
        if "Industry" in dataframe.columns:
            industries = dataframe["Industry"].dropna().unique().tolist()
            selected_industries = st.multiselect(
            "Industry",
            industries,
            default=[],
            key="industry_multiselect"
            )
            if selected_industries:
                dataframe = dataframe[dataframe["Industry"].isin(selected_industries)]
        st.dataframe(dataframe, use_container_width=True)

elif option == "Portfolio":
    st.header("Portfolio")
    st.write("Track your portfolio holdings and performance.")
    st.dataframe({
        "Asset": ["AAPL", "TSLA", "MSFT"],
        "Shares": [50, 20, 30],
        "Current Value": [9500, 14400, 9000]
    })
    # Apply theme change (Streamlit does not support dynamic theme switching at runtime)
    # But we can show a message and optionally write the config file for the user


elif option == "Settings":
    st.header("Settings")
    st.write("Configure your account and preferences here.")
    st.text_input("API Key")
    theme = st.selectbox("Theme", ["Light", "Dark"], key="theme_select")
    config_path = os.path.join(".streamlit", "config.toml")

    if st.button("Apply Theme"):
        theme_config = (
            '[theme]\n'
            f'base = "{theme.lower()}"\n'
        )
        os.makedirs(".streamlit", exist_ok=True)
        with open(config_path, "w") as f:
            f.write(theme_config)
        st.success(f"{theme} theme applied! Please reload the app to see changes.")
    st.info(
        "Theme changes will apply after you reload the app. "
        "To use a dark or light theme, set your preference in `.streamlit/config.toml`."
    )