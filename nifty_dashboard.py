import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta

# Set up Streamlit page
st.set_page_config(layout="wide")

# Cache function 
@st.cache_data
def fetch_stock_data(ticker, period, interval):
    try:
        end_date = datetime.now()
        if period == '1wk':
            start_date = end_date - timedelta(days=7)
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)

        if data.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def process_data(data):
    if not data.empty:
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('Asia/Kolkata')  # Set to IST
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data):
    if data.empty or len(data) < 2:
        st.warning("Not enough data to calculate metrics.")
        return 0, 0, 0, 0, 0, 0
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce').fillna(0)

    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()

    return last_close, change, pct_change, high, low, volume


def add_technical_indicators(data, sma_period, ema_period):
    if 'Close' not in data.columns:
        st.warning("The 'Close' column is missing from the data.")
        return data
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce').fillna(method='ffill')

    if len(data) < max(sma_period, ema_period):
        st.warning(f"Not enough data to calculate SMA ({sma_period}) or EMA ({ema_period}).")
        return data

    if data['Close'].isnull().all():
        st.warning("The 'Close' column contains only NaN values. Cannot calculate indicators.")
        return data
    try:
        data['SMA'] = ta.trend.sma_indicator(data['Close'], window=sma_period)
        data['EMA'] = ta.trend.ema_indicator(data['Close'], window=ema_period)
    except Exception as e:
        st.error(f"Error calculating SMA/EMA: {e}")

    return data


st.markdown("<h1 style='text-align: center; color: blue;'>Real-Time Indian Stock Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', '^NSEI') 
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
sma_period = st.sidebar.number_input('SMA Period', min_value=5, max_value=100, value=20)
ema_period = st.sidebar.number_input('EMA Period', min_value=5, max_value=100, value=20)
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA', 'EMA'])

interval_mapping = {
    '1d': '5m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

if st.sidebar.button('Update'):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    data = process_data(data)
    data = add_technical_indicators(data, sma_period, ema_period)

    if not data.empty:
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)

        # Display metrics
        st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} INR", delta=f"{change:.2f} ({pct_change:.2f}%)")
        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{high:.2f} INR")
        col2.metric("Low", f"{low:.2f} INR")
        col3.metric("Volume", f"{volume:,}")

        # Plotting the chart
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        for indicator in indicators:
            if indicator == 'SMA':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA'], name=f'SMA {sma_period}'))
            elif indicator == 'EMA':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA'], name=f'EMA {ema_period}'))

        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                          xaxis_title='Time (IST)',
                          yaxis_title='Price (INR)',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(100))

        st.subheader('Technical Indicators')
        st.dataframe(data[['Datetime', 'SMA', 'EMA']])
    else:
        st.error(f"No data available for {ticker}.")

st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '5m')

    if not real_time_data.empty:  # Ensure data is not empty
        real_time_data = process_data(real_time_data)
        
        if 'Close' in real_time_data.columns and 'Open' in real_time_data.columns:
            if not real_time_data['Close'].empty and not pd.isnull(real_time_data['Close'].iloc[-1]):
                last_price = real_time_data['Close'].iloc[-1]
            else:
                last_price = 0

            if not real_time_data['Open'].empty and not pd.isnull(real_time_data['Open'].iloc[0]):
                open_price = real_time_data['Open'].iloc[0]
            else:
                open_price = 0

            change = last_price - open_price
            pct_change = (change / open_price) * 100 if open_price != 0 else 0
            
            st.sidebar.metric(f"{symbol}", f"{last_price:.2f} INR", f"{change:.2f} ({pct_change:.2f}%)")
        else:
            st.sidebar.warning(f"Missing 'Close' or 'Open' data for {symbol}.")
    else:
        st.sidebar.warning(f"No data available for {symbol} at the moment.")

if not data.empty:
    st.subheader('Historical Data')
    st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(100))
    st.subheader('Technical Indicators')
    if 'SMA' in data.columns or 'EMA' in data.columns:
        st.dataframe(data[['Datetime', 'SMA', 'EMA']])
    else:
        st.warning("No technical indicators available.")
else:
    st.error(f"No data available for {ticker}.")


st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides Indian stock data and technical indicators for various time periods. Use the sidebar to customize your view.')
