import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import numpy as np

class Finance:
    def __init__(self):
        style.use('ggplot')
        self.start = dt.datetime(2015, 1, 1)
        self.end = dt.datetime.now()
        self.df = None

    def get_stock_price(self, ticker):
        """
        Fetch stock price data using Yahoo Finance API
        
        Parameters:
        ticker (str): Stock ticker symbol
        
        Returns:
        pandas.DataFrame: DataFrame containing stock price data
        """
        try:
            self.df = yf.download(ticker, self.start, self.end)
            # Reset index to make Date a column
            self.df.reset_index(inplace=True)
            return self.df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_moving_avg(self, ticker, window=5, price_col='Adj Close'):
        """
        Calculate and plot moving average
        
        Parameters:
        ticker (str): Stock ticker symbol
        window (int): Window size for moving average calculation
        price_col (str): Column name to use for price data
        
        Returns:
        fig, ax: Matplotlib figure and axis objects
        action (str): Trading signal based on moving average
        """
        # Get stock data if not already loaded
        if self.df is None:
            self.df = self.get_stock_price(ticker)
        
        # Make sure price_col exists in the dataframe
        if price_col not in self.df.columns and 'Close' in self.df.columns:
            price_col = 'Close'  # Fallback to Close if Adj Close doesn't exist
        
        # Calculate moving average
        self.df[f'{window} MA'] = self.df[price_col].rolling(window=window, min_periods=1).mean()
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df['Date'], self.df[price_col], label='Stock Price', color='#1E88E5')
        ax.plot(self.df['Date'], self.df[f'{window} MA'], label=f'{window}-day MA', color='#4CAF50', linestyle='--')
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Determine latest signal (BUY/HOLD or SELL)
        if self.df[price_col].iloc[-1] > self.df[f'{window} MA'].iloc[-1]:
            action = 'BUY/HOLD'
        else:
            action = 'SELL'
        
        ax.set_title(f"{ticker} with {window}-day Moving Average\nLatest Signal: {action}")
        plt.tight_layout()
        
        return fig, ax, action
    
    def calculate_technical_indicators(self, price_col='Adj Close'):
        """
        Calculate various technical indicators
        
        Parameters:
        price_col (str): Column name to use for price data
        
        Returns:
        pandas.DataFrame: DataFrame with added technical indicators
        """
        if self.df is None or self.df.empty:
            return self.df
        
        # Make sure price_col exists in the dataframe
        if price_col not in self.df.columns and 'Close' in self.df.columns:
            price_col = 'Close'  # Fallback to Close if Adj Close doesn't exist
        
        # Calculate daily returns
        self.df['Daily_Return'] = self.df[price_col].pct_change() * 100
        
        # Calculate moving averages
        for period in [5, 10, 20, 50, 200]:
            self.df[f'MA{period}'] = self.df[price_col].rolling(window=period, min_periods=1).mean()
        
        # Calculate exponential moving averages
        for period in [12, 26, 50]:
            self.df[f'EMA{period}'] = self.df[price_col].ewm(span=period, adjust=False).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = self.df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['Signal_Line']
        
        # Calculate Bollinger Bands
        window = 20
        self.df[f'MA{window}'] = self.df[price_col].rolling(window=window).mean()
        self.df['BB_Std'] = self.df[price_col].rolling(window=window).std()
        self.df['BB_Upper'] = self.df[f'MA{window}'] + (self.df['BB_Std'] * 2)
        self.df['BB_Lower'] = self.df[f'MA{window}'] - (self.df['BB_Std'] * 2)
        
        # Calculate ATR (Average True Range) - volatility indicator
        high = self.df['High'] if 'High' in self.df.columns else self.df[price_col]
        low = self.df['Low'] if 'Low' in self.df.columns else self.df[price_col]
        close = self.df[price_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        
        return self.df
    
    def get_trading_signals(self):
        """
        Generate trading signals based on technical indicators
        
        Returns:
        dict: Dictionary containing various trading signals
        """
        if self.df is None or self.df.empty:
            return {}
        
        signals = {}
        
        # Moving Average signals
        signals['MA_Signal'] = 'BUY' if self.df['MA5'].iloc[-1] > self.df['MA20'].iloc[-1] else 'SELL'
        
        # RSI signals
        current_rsi = self.df['RSI'].iloc[-1]
        if current_rsi < 30:
            signals['RSI_Signal'] = 'OVERSOLD (BUY)'
        elif current_rsi > 70:
            signals['RSI_Signal'] = 'OVERBOUGHT (SELL)'
        else:
            signals['RSI_Signal'] = 'NEUTRAL'
        
        # MACD signals
        if self.df['MACD'].iloc[-1] > self.df['Signal_Line'].iloc[-1]:
            signals['MACD_Signal'] = 'BULLISH'
        else:
            signals['MACD_Signal'] = 'BEARISH'
        
        # Bollinger Bands signals
        last_price = self.df['Adj Close'].iloc[-1] if 'Adj Close' in self.df.columns else self.df['Close'].iloc[-1]
        upper_band = self.df['BB_Upper'].iloc[-1]
        lower_band = self.df['BB_Lower'].iloc[-1]
        
        if last_price > upper_band:
            signals['BB_Signal'] = 'OVERBOUGHT (SELL)'
        elif last_price < lower_band:
            signals['BB_Signal'] = 'OVERSOLD (BUY)'
        else:
            signals['BB_Signal'] = 'NEUTRAL'
        
        # Calculate overall signal
        buy_signals = len([s for s in signals.values() if 'BUY' in s or 'BULLISH' in s])
        sell_signals = len([s for s in signals.values() if 'SELL' in s or 'BEARISH' in s])
        
        if buy_signals > sell_signals:
            signals['Overall_Signal'] = 'BUY'
        elif sell_signals > buy_signals:
            signals['Overall_Signal'] = 'SELL'
        else:
            signals['Overall_Signal'] = 'HOLD'
        
        return signals
    
    def calculate_performance_metrics(self, price_col='Adj Close'):
        """
        Calculate various performance metrics for the stock
        
        Parameters:
        price_col (str): Column name to use for price data
        
        Returns:
        dict: Dictionary containing performance metrics
        """
        if self.df is None or self.df.empty:
            return {}
        
        # Make sure price_col exists in the dataframe
        if price_col not in self.df.columns and 'Close' in self.df.columns:
            price_col = 'Close'  # Fallback to Close if Adj Close doesn't exist
        
        metrics = {}
        
        # Basic statistics
        metrics['Mean_Price'] = self.df[price_col].mean()
        metrics['Max_Price'] = self.df[price_col].max()
        metrics['Min_Price'] = self.df[price_col].min()
        metrics['Price_Std'] = self.df[price_col].std()
        metrics['Latest_Price'] = self.df[price_col].iloc[-1]
        
        # Calculate returns
        self.df['Daily_Return'] = self.df[price_col].pct_change() * 100
        metrics['Mean_Daily_Return'] = self.df['Daily_Return'].mean()
        
        # Calculate period return
        first_price = self.df[price_col].iloc[0]
        last_price = self.df[price_col].iloc[-1]
        metrics['Period_Return'] = ((last_price / first_price) - 1) * 100
        
        # Calculate annualized metrics
        trading_days_per_year = 252
        metrics['Annualized_Return'] = metrics['Mean_Daily_Return'] * trading_days_per_year
        metrics['Annualized_Volatility'] = self.df['Daily_Return'].std() * np.sqrt(trading_days_per_year)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0% for simplicity)
        if metrics['Annualized_Volatility'] != 0:
            metrics['Sharpe_Ratio'] = metrics['Annualized_Return'] / metrics['Annualized_Volatility']
        else:
            metrics['Sharpe_Ratio'] = 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + self.df['Daily_Return'] / 100).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        metrics['Max_Drawdown'] = drawdown.min() * 100
        
        return metrics