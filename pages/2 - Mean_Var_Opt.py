import streamlit as st
import pandas as pd
import numpy as np
from psycopg2 import sql
import psycopg2, psycopg2.extras
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import scipy.optimize as sc

st.set_page_config(
    page_title="Mean-Variance Optimization Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource(ttl=600)
def get_database_session():
    # Create a database session object that points to the URL.
    conn = psycopg2.connect(**st.secrets.db_credentials)
    return conn

def get_stock_id_symbol_lookup_df(conn):
    cursor = conn.cursor()
    cursor.execute(sql.SQL("select id, symbol from stock;"))
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    return df

def get_data(conn, selected_id, no_years, end_date, stock_id_lookup_df):
    selected_id = tuple(selected_id)
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=no_years)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Create a cursor object to execute SQL statements
    cursor = conn.cursor()
    cursor.execute(
        sql.SQL("select stock_id, date, close from stock_price where date >= %s AND date <= %s AND stock_id IN %s order by date asc;"),
        [start_date, end_date, selected_id]
    )
    data = cursor.fetchall()
    data = pd.DataFrame(data, columns=['stock_id', 'dt', 'close'])
    data = data.pivot(index='dt', columns='stock_id', values='close')
    data = data.rename(columns=lambda x: stock_id_lookup_df[stock_id_lookup_df['id'] == x]['symbol'].values[0])
    if 1 <= data.isna().sum().sum() <= 10:
        data = data.fillna(method='ffill')
    else:
        data = data.dropna(axis=1)
    
    data = data.pct_change()
    data = data.dropna()
    return data

def ret_graph(data):
    num_stocks = data.shape[1]
    num_cols = 4
    num_rows = (num_stocks // num_cols + 1) if num_stocks % num_cols != 0 else num_stocks // num_cols
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=data.columns, shared_xaxes=True, x_title='Daily Return Distribution')
    for i, ticker in enumerate(data.columns):
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1
        fig.add_trace(px.histogram(data[ticker], x=ticker).data[0], row=row, col=col)
    return fig

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(252)
    return returns, std

#Compute negative Sharpe ratio for minimization
def neg_SR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    p_ret, p_std = portfolio_performance(weights, meanReturns, covMatrix)
    return -(p_ret - riskFreeRate) / p_std

#Lagrangian optimization: maximize Sharpe ratio
def max_SR(meanReturns, covMatrix, riskFreeRate = 0, shortSale = False):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(numAssets))
    if shortSale == True:
        bounds = tuple((-1,1) for asset in range(numAssets))
    result = sc.minimize(neg_SR, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#Lagrangian optimization: minimize standard deviation
def p_std(weights, meanReturns, covMatrix):
    return portfolio_performance(weights, meanReturns, covMatrix)[1]
def min_std(meanReturns, covMatrix, shortSale = False):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(numAssets))
    if shortSale == True:
        bounds = tuple((-1,1) for asset in range(numAssets))
    result = sc.minimize(p_std, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#Creating the efficient frontier
def p_ret(weights, meanReturns, covMatrix):
    return portfolio_performance(weights, meanReturns, covMatrix)[0]

def efficient_frontier(meanReturns, covMatrix, returnTarget, shortSale = False):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: p_ret(x, meanReturns, covMatrix) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(numAssets))
    if shortSale == True:
        bounds = tuple((-1,1) for asset in range(numAssets))
    result = sc.minimize(p_std, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#Use a return function that could be called by a graphing package like Plotly/Dash combination.
def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, shortSale=False):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = max_SR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolio_performance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=data.columns, columns=['allocation'])
    
    # Min Volatility Portfolio
    minVol_Portfolio = min_std(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolio_performance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=data.columns, columns=['allocation'])

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(start=minVol_returns, stop=maxSR_returns, num = 20)
    for target in targetReturns:
        efficientList.append(efficient_frontier(meanReturns, covMatrix, target)['fun'])

    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

#visualize efficient frontier
def EF_graph(meanReturns, covMatrix, riskFreeRate=0, shortSale=False):
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, shortSale)
    fig = px.line(x=efficientList, y=targetReturns, title='Efficient Frontier', labels={'x': 'Standard Deviation', 'y': 'Returns'})
    fig.add_scatter(x=[maxSR_std], y=[maxSR_returns], mode='markers', name='Max Sharpe')
    fig.add_scatter(x=[minVol_std], y=[minVol_returns], mode='markers', name='Min Vol')
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Max Sharpe', 'Min Volatility'], specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig2.add_trace(px.pie(maxSR_allocation, values='allocation', names=maxSR_allocation.index).data[0], row=1, col=1)
    fig2.add_trace(px.pie(minVol_allocation, values='allocation', names=minVol_allocation.index).data[0], row=1, col=2)
    return fig, fig2

st.header('Mean-Variance Optimization')
with st.spinner('Connecting to Database...'):
    conn = get_database_session()
    stock_id_lookup_df = get_stock_id_symbol_lookup_df(conn)

tickers_list = stock_id_lookup_df['symbol'].tolist()
selected_tickers = st.multiselect(label='Select Tickers', options=tickers_list, default=['AAPL', 'MSFT', 'AMZN', 'GOOGL'])
no_years = st.sidebar.slider(label='Number of Years', min_value=1, max_value=10, value=10)
end_date = st.sidebar.date_input(label='End Date (Latest: 31st Dec, 2022)', value=datetime(2022, 12, 31))

#Retrieve data from database
button_clicked = st.button(label='Retrieve Data', type='primary')
if button_clicked:
    with st.spinner(text='Retrieving Data...'):
        selected_id = stock_id_lookup_df[stock_id_lookup_df['symbol'].isin(selected_tickers)]['id'].tolist()
        data = get_data(conn, selected_id, no_years, end_date, stock_id_lookup_df)
        st.plotly_chart(ret_graph(data), use_container_width=True)
        mean_returns_list = np.array([data[ticker].mean() for ticker in data.columns])
        cov_matrix = data.cov()
        st.plotly_chart(EF_graph(mean_returns_list, cov_matrix)[0], use_container_width=True)
        st.plotly_chart(EF_graph(mean_returns_list, cov_matrix)[1], use_container_width=True)