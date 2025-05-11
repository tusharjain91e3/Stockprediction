import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from PythonFinance import Finance  # Import the Finance class from your other file

# Streamlit UI configuration
st.set_page_config(
    page_title="📈 StockGenie - AI Powered Stock Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 26px;
        font-weight: 500;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: red;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color:black;
    }
    .prediction-card {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>🧠 StockGenie</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Predict and visualize stock prices with advanced AI models. Make smarter investment decisions with data-driven insights.</div>", unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("📊 Configuration")
    
    # User input
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.NS):", "AAPL")
    
    # Date range selection with sensible defaults
    default_start = pd.to_datetime("today") - timedelta(days=365)  # 1 year ago
    default_end = pd.to_datetime("today")
    
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)
    
    # Model selection
    prediction_model = st.selectbox(
        "Select Prediction Model", 
        ["Linear Regression", "ARIMA"]
    )
    
    prediction_days = st.slider("Prediction Days", min_value=3, max_value=30, value=7)
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_ma = st.checkbox("Show Moving Averages", value=True)
    ma_periods = st.multiselect("MA Periods", [20, 50, 100, 200], default=[20])
    
    show_volume = st.checkbox("Show Volume Chart", value=True)
    
    # Add a button to trigger analysis
    analyze_button = st.button("🔍 Analyze Stock", type="primary")

# Main content
if ticker and analyze_button:
    # Show loading spinner while fetching data
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"No data found for {ticker}. Please check the ticker symbol and try again.")
                st.stop()
                
            # Make a copy of the dataframe to avoid SettingWithCopyWarning
            stock_data = df.copy()
            price_column = "Adj Close" if "Adj Close" in stock_data.columns else "Close"
            
            # Display success message
            st.success(f"✅ Successfully fetched data for {ticker} from {start_date} to {end_date}")
            
            # Display company info
            try:
                ticker_info = yf.Ticker(ticker)
                company_name = ticker_info.info.get('longName', ticker)
                sector = ticker_info.info.get('sector', 'N/A')
                industry = ticker_info.info.get('industry', 'N/A')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Company", company_name)
                with col2:
                    st.metric("Sector", sector)
                with col3:
                    st.metric("Industry", industry)
            except:
                st.info(f"Basic analysis for {ticker}")
            
            # Latest price information
            last_price = float(stock_data[price_column].iloc[-1])  # Convert to float to fix the error
            prev_price = float(stock_data[price_column].iloc[-2])
            price_change = last_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                # Make sure last_price is a single float value, not a Series
                last_price_value = float(last_price) if hasattr(last_price, '__iter__') else last_price
                st.metric(f"Latest {price_column} Price", f"${last_price_value:.2f}")
            with col2:
                # Make sure price_change is a single float value
                price_change_value = float(price_change) if hasattr(price_change, '__iter__') else price_change
                price_change_pct_value = float(price_change_pct) if hasattr(price_change_pct, '__iter__') else price_change_pct
                st.metric("Price Change", f"${price_change_value:.2f}", f"{price_change_pct_value:.2f}%")
            with col3:
                # Make sure volume is a single value
                volume_value = int(stock_data['Volume'].iloc[-1]) if hasattr(stock_data['Volume'].iloc[-1], '__iter__') else stock_data['Volume'].iloc[-1]
                st.metric("Trading Volume", f"{volume_value:,}")
            
            # Calculate moving averages if requested
            if show_ma:
                for period in ma_periods:
                    stock_data[f'MA{period}'] = stock_data[price_column].rolling(window=period).mean()
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🔮 Price Prediction", "📊 Technical Analysis", "📝 Summary"])
            
            # Tab 1: Price Chart
            with tab1:
                st.markdown("<h2 class='sub-header'>Historical Price Chart</h2>", unsafe_allow_html=True)
                
                # Create interactive plot with Plotly
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data[price_column],
                    mode='lines',
                    name=f'{price_column} Price',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # Add moving averages
                if show_ma:
                    colors = ['#FFC107', '#4CAF50', '#FF5722', '#9C27B0']
                    for i, period in enumerate(ma_periods):
                        if f'MA{period}' in stock_data.columns:
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data[f'MA{period}'],
                                mode='lines',
                                name=f'{period}-day MA',
                                line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
                            ))
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Price History',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add volume chart if requested
                if show_volume:
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=stock_data.index,
                        y=stock_data['Volume'],
                        name='Volume',
                        marker=dict(color='rgba(30, 136, 229, 0.5)')
                    ))
                    
                    volume_fig.update_layout(
                        title='Trading Volume',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        height=300,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(volume_fig, use_container_width=True)
            
            # Tab 2: Price Prediction
            with tab2:
                st.markdown("<h2 class='sub-header'>AI-Powered Price Prediction</h2>", unsafe_allow_html=True)
                
                # Prepare data for prediction
                stock_data.reset_index(inplace=True)
                stock_data['Date_Ordinal'] = pd.to_datetime(stock_data['Date']).map(pd.Timestamp.toordinal)
                
                # Generate future dates
                last_date = stock_data['Date'].iloc[-1]
                future_dates = pd.date_range(last_date, periods=prediction_days+1, freq='D')[1:]
                future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                
                # Prediction based on selected model
                if prediction_model == "Linear Regression":
                    # Train linear regression model
                    X = stock_data['Date_Ordinal'].values.reshape(-1, 1)
                    y = stock_data[price_column].values.reshape(-1, 1)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Make predictions
                    future_preds = model.predict(future_ordinal)
                    
                    # Calculate confidence (just for demonstration, simple approach)
                    confidence = f"{abs(model.score(X, y) * 100):.1f}%"
                    
                    model_description = """
                    Linear Regression model predicts future prices based on the linear trend in historical data.
                    It works best for stocks with consistent trends but may not capture market volatility.
                    """
                    
                elif prediction_model == "ARIMA":
                    try:
                        # Prepare data for ARIMA
                        y = stock_data[price_column].values
                        
                        # Fit ARIMA model
                        model = ARIMA(y, order=(5,1,0))
                        model_fit = model.fit()
                        
                        # Forecast
                        forecast = model_fit.forecast(steps=prediction_days)
                        future_preds = forecast.reshape(-1, 1)
                        
                        # Confidence
                        confidence = "Based on time series analysis"
                        
                        model_description = """
                        ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting model
                        that captures temporal structures in the data. It's better at handling market volatility
                        compared to simple regression.
                        """
                    except Exception as e:
                        st.error(f"Error in ARIMA modeling: {e}")
                        st.info("Falling back to Linear Regression model")
                        
                        # Fallback to Linear Regression
                        X = stock_data['Date_Ordinal'].values.reshape(-1, 1)
                        y = stock_data[price_column].values.reshape(-1, 1)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        future_preds = model.predict(future_ordinal)
                        confidence = f"{abs(model.score(X, y) * 100):.1f}%"
                        model_description = "(Fallback to Linear Regression model due to ARIMA error)"
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_preds.flatten()
                })
                
                # Display prediction plot
                pred_fig = go.Figure()
                
                # Historical data
                pred_fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data[price_column],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # Prediction
                pred_fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Predicted Price'],
                    mode='lines+markers',
                    name='Predicted Prices',
                    line=dict(color='#4CAF50', width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                # Update layout
                pred_fig.update_layout(
                    title=f'{ticker} Price Prediction (Next {prediction_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    template='plotly_white'
                )
                
                # Add confidence band (very simplified for demonstration)
                if prediction_model == "Linear Regression":
                    # Add a simple confidence band (+/- 5%)
                    upper_bound = pred_df['Predicted Price'] * 1.05
                    lower_bound = pred_df['Predicted Price'] * 0.95
                    
                    pred_fig.add_trace(go.Scatter(
                        x=pred_df['Date'],
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    pred_fig.add_trace(go.Scatter(
                        x=pred_df['Date'],
                        y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(76, 175, 80, 0.2)',
                        name='Confidence Range'
                    ))
                
                st.plotly_chart(pred_fig, use_container_width=True)
                
                # Display prediction details
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown(f"### Prediction Details ({prediction_model})")
                st.markdown(model_description)
                st.markdown(f"**Model Confidence:** {confidence}")
                
                # Calculate expected trend
                first_pred = pred_df['Predicted Price'].iloc[0]
                last_pred = pred_df['Predicted Price'].iloc[-1]
                trend_pct = ((last_pred - first_pred) / first_pred) * 100
                
                if trend_pct > 5:
                    trend_message = f"📈 Strong Upward Trend: +{trend_pct:.2f}%"
                    recommendation = "Consider BUY if aligns with your investment strategy"
                elif trend_pct > 0:
                    trend_message = f"↗️ Slight Upward Trend: +{trend_pct:.2f}%"
                    recommendation = "Consider HOLD/BUY if aligns with your investment strategy"
                elif trend_pct > -5:
                    trend_message = f"↘️ Slight Downward Trend: {trend_pct:.2f}%"
                    recommendation = "HOLD if already invested, monitor closely"
                else:
                    trend_message = f"📉 Strong Downward Trend: {trend_pct:.2f}%"
                    recommendation = "Consider SELL or wait for trend reversal"
                
                st.markdown(f"**Expected Trend:** {trend_message}")
                st.markdown(f"**AI Suggestion:** {recommendation}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display prediction table
                st.markdown("### Detailed Price Predictions")
                # Format the predicted price column with a safer approach
                pred_df['Predicted Price'] = pred_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
                st.dataframe(pred_df, use_container_width=True)
                
                # Disclaimer
                st.info("""
                **Disclaimer:** These predictions are based on historical data and should not be
                considered as financial advice. Always conduct your own research before making
                investment decisions.
                """)
            
            # Tab 3: Technical Analysis
            with tab3:
                st.markdown("<h2 class='sub-header'>Technical Analysis</h2>", unsafe_allow_html=True)
                
                # Add technical analysis from pythonfinance.py
                st.subheader("Moving Average Analysis")
                
                # Create new figure for MA analysis
                ma_fig, ax = plt.subplots(figsize=(10, 6))
                
                # Use Finance class from pythonfinance.py
                finance = Finance()
                
                # Customize the date range to match user selection
                finance.start = start_date
                finance.end = end_date
                
                # Get stock data (we'll use our already fetched data)
                finance.df = stock_data.copy()
                
                # Calculate 5-day moving average
                finance.df['1 ma'] = finance.df[price_column].rolling(window=5, min_periods=1).mean()
                
                # Plot data using matplotlib
                ax.plot(finance.df['Date'], finance.df[price_column], label='Stock Price', color='yellow')
                ax.plot(finance.df['Date'], finance.df['1 ma'], label='5-day MA', color='cyan')
                ax.legend()
                
                # Simple latest day action (copied from pythonfinance.py)
                if finance.df[price_column].iloc[-1] > finance.df['1 ma'].iloc[-1]:
                    action = 'BUY/HOLD'
                else:
                    action = 'SELL'
                
                plt.title(f"Ticker Symbol: {ticker}\nLatest Signal: {action}")
                plt.tight_layout()
                
                st.pyplot(ma_fig)
                
                # Additional technical indicators
                st.subheader("Additional Technical Indicators")
                
                # Calculate RSI (Relative Strength Index)
                delta = stock_data[price_column].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                stock_data['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD (Moving Average Convergence Divergence)
                stock_data['EMA12'] = stock_data[price_column].ewm(span=12, adjust=False).mean()
                stock_data['EMA26'] = stock_data[price_column].ewm(span=26, adjust=False).mean()
                stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
                stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
                
                # Create indicators plot
                ind_fig = go.Figure()
                
                # RSI subplot
                ind_fig = go.Figure(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1.5)
                ))
                
                # Add reference lines for RSI
                ind_fig.add_shape(
                    type="line",
                    x0=stock_data['Date'].iloc[0],
                    y0=70,
                    x1=stock_data['Date'].iloc[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash")
                )
                
                ind_fig.add_shape(
                    type="line",
                    x0=stock_data['Date'].iloc[0],
                    y0=30,
                    x1=stock_data['Date'].iloc[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash")
                )
                
                ind_fig.update_layout(
                    title='Relative Strength Index (RSI)',
                    xaxis_title='Date',
                    yaxis_title='RSI',
                    yaxis=dict(range=[0, 100]),
                    height=300
                )
                
                st.plotly_chart(ind_fig, use_container_width=True)
                
                # MACD plot
                macd_fig = go.Figure()
                
                # MACD Line
                macd_fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                ))
                
                # Signal Line
                macd_fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['Signal_Line'],
                    mode='lines',
                    name='Signal Line',
                    line=dict(color='red', width=1.5)
                ))
                
                # MACD Histogram
                macd_fig.add_trace(go.Bar(
                    x=stock_data['Date'],
                    y=stock_data['MACD'] - stock_data['Signal_Line'],
                    name='Histogram',
                    marker=dict(color='green')
                ))
                
                macd_fig.update_layout(
                    title='MACD (Moving Average Convergence Divergence)',
                    xaxis_title='Date',
                    yaxis_title='MACD',
                    height=300
                )
                
                st.plotly_chart(macd_fig, use_container_width=True)
                
                # Trading signals based on indicators
                st.subheader("Trading Signals")
                
                current_rsi = stock_data['RSI'].iloc[-1]
                rsi_signal = "Oversold (Potential Buy)" if current_rsi < 30 else "Overbought (Potential Sell)" if current_rsi > 70 else "Neutral"
                
                macd = stock_data['MACD'].iloc[-1]
                signal_line = stock_data['Signal_Line'].iloc[-1]
                macd_signal = "Bullish (MACD > Signal Line)" if macd > signal_line else "Bearish (MACD < Signal Line)"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RSI Value", f"{current_rsi:.2f}", rsi_signal)
                with col2:
                    st.metric("MACD Signal", macd_signal, f"{macd - signal_line:.4f}")
            
            # Tab 4: Summary
            with tab4:
                st.markdown("<h2 class='sub-header'>Investment Summary</h2>", unsafe_allow_html=True)
                
                # Calculate key statistics
                mean_price = stock_data[price_column].mean()
                max_price = stock_data[price_column].max()
                min_price = stock_data[price_column].min()
                price_std = stock_data[price_column].std()
                
                # Calculate returns
                stock_data['Daily_Return'] = stock_data[price_column].pct_change() * 100
                mean_daily_return = stock_data['Daily_Return'].mean()
                
                # Annualized metrics
                trading_days_per_year = 252
                annualized_return = mean_daily_return * trading_days_per_year
                annualized_volatility = stock_data['Daily_Return'].std() * np.sqrt(trading_days_per_year)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
                
                # Display statistics
                st.markdown("### Key Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"${mean_price:.2f}")
                    st.metric("Maximum Price", f"${max_price:.2f}")
                    st.metric("Minimum Price", f"${min_price:.2f}")
                with col2:
                    st.metric("Price Volatility", f"${price_std:.2f}")
                    st.metric("Daily Return", f"{mean_daily_return:.2f}%")
                    st.metric("Annualized Return", f"{annualized_return:.2f}%")
                with col3:
                    st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    period_return = ((last_price / stock_data[price_column].iloc[0]) - 1) * 100
                    st.metric("Period Return", f"{period_return:.2f}%")
                
                # Risk assessment based on volatility
                st.markdown("### Risk Assessment")
                
                if annualized_volatility < 15:
                    risk_level = "Low"
                    risk_description = "This stock shows relatively low volatility, suggesting lower risk but potentially lower returns."
                elif annualized_volatility < 25:
                    risk_level = "Moderate"
                    risk_description = "This stock shows moderate volatility, balancing risk and potential returns."
                else:
                    risk_level = "High"
                    risk_description = "This stock shows high volatility, suggesting higher risk with potential for higher returns."
                
                st.info(f"**Risk Level: {risk_level}**\n\n{risk_description}")
                
                # Overall assessment
                st.markdown("### Overall Assessment")
                
                # Combine signals for overall assessment
                signals = []
                if period_return > 10:
                    signals.append("Strong historical performance")
                
                if trend_pct > 0:
                    signals.append("Positive forecast trend")
                
                if current_rsi < 40:
                    signals.append("RSI indicates potential buying opportunity")
                elif current_rsi > 60:
                    signals.append("RSI indicates potential selling opportunity")
                
                if macd > signal_line:
                    signals.append("MACD indicates bullish momentum")
                else:
                    signals.append("MACD indicates bearish momentum")
                
                # Overall recommendation
                if len([s for s in signals if "bullish" in s.lower() or "buying" in s.lower() or "positive" in s.lower() or "strong" in s.lower()]) > len(signals)/2:
                    overall = "Positive Outlook"
                elif len([s for s in signals if "bearish" in s.lower() or "selling" in s.lower() or "negative" in s.lower()]) > len(signals)/2:
                    overall = "Negative Outlook"
                else:
                    overall = "Neutral Outlook"
                
                st.markdown(f"#### {overall}")
                for signal in signals:
                    st.markdown(f"- {signal}")
                
                # Disclaimer
                st.warning("""
                **Disclaimer:** This analysis is based on historical data and mathematical models.
                Past performance is not indicative of future results. This tool should be used for
                educational purposes only and not as financial advice. Always consult with a financial
                advisor before making investment decisions.
                """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")
else:
    # Display welcome message if no analysis has been performed yet
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### Welcome to StockGenie! 🧙‍♂️")
    st.markdown("""
    StockGenie uses artificial intelligence to analyze stock data and provide predictions and insights.
    
    To get started:
    1. Enter a stock ticker symbol in the sidebar (e.g., AAPL for Apple Inc.)
    2. Select your desired date range
    3. Choose a prediction model
    4. Click "Analyze Stock" to view the results
    
    StockGenie supports many stock exchanges worldwide. For Indian stocks, add the .NS suffix (e.g., INFY.NS).
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show some example tickers
    st.markdown("### Popular Stock Tickers")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**US Tech:**")
        st.markdown("- AAPL (Apple)")
        st.markdown("- MSFT (Microsoft)")
        st.markdown("- GOOGL (Alphabet)")
    with col2:
        st.markdown("**US Others:**")
        st.markdown("- AMZN (Amazon)")
        st.markdown("- JPM (JPMorgan)")
        st.markdown("- KO (Coca-Cola)")
    with col3:
        st.markdown("**India:**")
        st.markdown("- INFY.NS (Infosys)")
        st.markdown("- RELIANCE.NS (Reliance)")
        st.markdown("- TATAMOTORS.NS (Tata Motors)")
    with col4:
        st.markdown("**ETFs:**")
        st.markdown("- SPY (S&P 500)")
        st.markdown("- QQQ (Nasdaq 100)")
        st.markdown("- VTI (Total Market)")