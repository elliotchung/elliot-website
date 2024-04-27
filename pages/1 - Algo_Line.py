import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import json
import numpy as np
import pandas as pd

from psycopg2 import sql
import psycopg2

st.set_page_config(
    page_title="Algo Line Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource(ttl=600)
def get_database_session():
    # Create a database session object that points to the URL.
    conn = psycopg2.connect(**st.secrets.db_credentials)
    return conn

def get_df(stock_id):
    conn = get_database_session()
    # Create a cursor object to execute SQL statements
    cursor = conn.cursor()
    cursor.execute(
        sql.SQL("select date, open, high, low, close, volume from stock_price where stock_id = %s order by date asc;"),
        [stock_id]
    )
    data = cursor.fetchall()

    # Some data wrangling to match required format
    COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
    COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
    columns = ['time','open','high','low','close','volume']
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    df['color'] = np.where(  df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear
    df['volume_avg_50'] = df['volume'].rolling(window=50).mean().reset_index(drop=True)

    cursor.close()
    return df

def get_stock_id_symbol_lookup_df():
    conn = get_database_session()
    cursor = conn.cursor()
    cursor.execute(sql.SQL("select id, symbol from stock;"))
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    return df

def get_trendline_df(stock_id):
    conn = get_database_session()
    # Create a cursor object to execute SQL statements
    cursor = conn.cursor()
    cursor.execute(
        sql.SQL("select start_idx, end_idx, breach_idx, gradient, y_intercept, num_touches from possible_trendline where stock_id = %s;"),
        [stock_id]
    )
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    return df

def filter_trendline_df(trendline_df, trendline_length=1, gradient_range=(-1, 1), min_touches=10):
    trendline_df = trendline_df[trendline_df.breach_idx - trendline_df.start_idx >= trendline_length]
    trendline_df = trendline_df[(trendline_df.gradient >= gradient_range[0]) & (trendline_df.gradient <= gradient_range[1])]
    trendline_df = trendline_df[trendline_df.num_touches >= min_touches]
    trendline_df.reset_index(drop=True, inplace=True)
    return trendline_df

def create_trendline_data_df(df, start, breach, gradient, y_intercept):
    x = np.arange(start, breach+1)
    y = gradient * x + y_intercept
    trendline_data_df = pd.DataFrame({'time': df.loc[start:breach, 'time'], 'value': y})
    trendline_data_df.reset_index(drop=True, inplace=True)
    trendline_data_df['time'] = pd.to_datetime(trendline_data_df['time'])
    trendline_data_df['time'] = trendline_data_df['time'].dt.strftime('%Y-%m-%d')
    return trendline_data_df

def combine_trendline_data_df(df, filtered_trendline_df):
    all_trendline_data_df_list = []
    for i in range(len(filtered_trendline_df)):
        trendline_data_df = create_trendline_data_df(df, filtered_trendline_df.loc[i, 'start_idx'], 
                                                     filtered_trendline_df.loc[i, 'breach_idx'], 
                                                     filtered_trendline_df.loc[i, 'gradient'], 
                                                     filtered_trendline_df.loc[i, 'y_intercept'])
        all_trendline_data_df_list.append(trendline_data_df)
    return all_trendline_data_df_list

def create_priceVolumeSeries(all_trendline_data_df_list, candles, volume, volume_avg_50, COLOR_BULL = 'rgba(38,166,154,0.9)', COLOR_BEAR = 'rgba(239,83,80,0.9)'):
    COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
    COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
    priceVolumeSeries = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR,
                "priceLineVisible": False,
                "lastValueVisible": False,
            }
        },
        {
            "type": 'Histogram',
            "data": volume,
            "options": {
                "color": '#26a69a',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceLineVisible": False,
                "lastValueVisible": False,
                "priceScaleId": "volume" # set as an overlay setting,
            }
            },
        {
            "type": 'Line',
            "data": volume_avg_50,
            "options": {
                "color": 'white',
                "lineStyle": 0,
                "lineWidth": 0.5,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "priceScaleId": "volume"
            }
        }
    ]
    for trendline_data_df in all_trendline_data_df_list:
        trendline_data_json = json.loads(trendline_data_df.to_json(orient='records'))
        priceVolumeSeries.append({
            "type": 'Line',
            "data": trendline_data_json,
            "options": {
                "color": 'rgba(255, 255, 255, 0.5)',
                "lineStyle": 0,
                "lineWidth": 1,
                "priceLineVisible": False,
                "lastValueVisible": False,
                "crosshairMarkerVisible": False,
            }
        })
    return priceVolumeSeries

ticker_lookup_df = get_stock_id_symbol_lookup_df()
ticker_select = st.selectbox('Select stock ticker', ticker_lookup_df['symbol'], index=3)
id = int(ticker_lookup_df[ticker_lookup_df['symbol'] == ticker_select]['id'].values[0])
df = get_df(id)
trendline_df = get_trendline_df(id)

chart_height = st.sidebar.slider("Chart height", 400, 1200, 800)

st.sidebar.subheader("Customise Trendline:")
trendline_length = st.sidebar.slider("Minimum trendline length", 1, 2000, 100)
gradient_threshold = st.sidebar.slider("Maximum gradient", -1., 1., 0., 0.01)
min_touches = st.sidebar.slider("Minimum touches", 1, 100, 10)

filtered_trendline_df = filter_trendline_df(trendline_df, trendline_length, gradient_threshold, min_touches)

if len(filtered_trendline_df) > 500:
    st.write("Too many trendlines, only displaying sampled 500")
    filtered_trendline_df = filtered_trendline_df.sample(500)
    filtered_trendline_df.reset_index(drop=True, inplace=True)

priceVolumeChartOptions = {
    "height": chart_height,
    "rightPriceScale": {
        "scaleMargins": {
            "top": 0.2,
            "bottom": 0.25,
        },
        "borderVisible": False,
    },
    "overlayPriceScales": {
        "scaleMargins": {
            "top": 0.8,
            "bottom": 0,
        }
    },
    "layout": {
        "background": {
            "type": 'solid',
            "color": '#131722'
        },
        "textColor": '#d1d4dc',
    },
    "grid": {
        "vertLines": {
            "color": 'rgba(42, 46, 57, 0)',
        },
        "horzLines": {
            "color": 'rgba(42, 46, 57, 0)',
        }
    }
}

candles_df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
volume_df = df[['time', 'volume']]
volume_avg_50 = df[['time', 'volume_avg_50']]
candles = json.loads(candles_df.to_json(orient = "records"))
volume = json.loads(volume_df.rename(columns={"volume": "value",}).to_json(orient = "records"))
volume_avg_50 = json.loads(volume_avg_50.rename(columns={"volume_avg_50": "value",}).to_json(orient = "records"))

try:
    trendline_data_arr = create_trendline_data_df(df, filtered_trendline_df.loc[0, 'start_idx'], 
                                                filtered_trendline_df.loc[0, 'breach_idx'], 
                                                filtered_trendline_df.loc[0, 'gradient'], 
                                                filtered_trendline_df.loc[0, 'y_intercept'])
    all_trendline_data_df_list = combine_trendline_data_df(df, filtered_trendline_df)

    # export to JSON format
    trendline = json.loads(trendline_data_arr.to_json(orient = "records"))

    priceVolumeSeries = create_priceVolumeSeries(all_trendline_data_df_list, candles, volume, volume_avg_50)

    st.write("Number of trendlines:", len(filtered_trendline_df))

    renderLightweightCharts([
        {
            "chart": priceVolumeChartOptions,
            "series": priceVolumeSeries
        }
    ], 'priceAndVolume')


except KeyError:
    st.write("No trendlines found")
    priceVolumeSeries = create_priceVolumeSeries([], candles, volume, volume_avg_50)
    renderLightweightCharts([
        {
            "chart": priceVolumeChartOptions,
            "series": priceVolumeSeries
        }
    ], 'priceAndVolume')