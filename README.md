# Stock Change Tracker - AI-Powered Trading Dashboard

A comprehensive stock analysis dashboard built with Streamlit that displays stock performance, technical analysis, and AI-powered trading signals using your existing MySQL database.

## 🚀 Features

### Home Dashboard
- **Stock Performance Table**: View all stocks ordered by change percentage (descending)
- **Earnings Highlights**: Stocks with earnings in the next 3 weeks are highlighted in green
- **Real-time Metrics**: Total stocks, gainers, losers, and average change percentage
- **Interactive Navigation**: Click any stock symbol to dive into detailed analysis

### Detailed Stock Analysis (3 Tabs)
1. **14-Day RSI Analysis**: 
   - Interactive RSI chart with overbought/oversold indicators
   - Current RSI value and trading signals
   - Historical RSI trend analysis

2. **Extreme RSI Analysis (>75)**:
   - Identifies periods when RSI exceeded 75 (extreme overbought)
   - Statistics on frequency and duration of extreme conditions
   - Table of all extreme overbought periods

3. **Neuro-Evolution Trading Signals**:
   - AI-powered trading signals using genetic algorithm
   - Buy/Sell signal visualization on price charts
   - Confidence levels for each signal
   - Recent trading signals table

## 🎯 Why Streamlit?

I chose **Streamlit** for this project because:

### 1. **Rapid Development**
- Build interactive web apps with minimal code
- No need for HTML, CSS, or JavaScript knowledge
- Perfect for data science and financial applications

### 2. **Built-in Data Visualization**
- Seamless integration with Plotly, Pandas, and NumPy
- Interactive charts and tables out of the box
- Easy-to-implement filtering and navigation

### 3. **Database Integration**
- Simple MySQL connection with SQLAlchemy
- Efficient caching with `@st.cache_data`
- Real-time data updates

### 4. **User Experience**
- Professional-looking interface with minimal setup
- Responsive design that works on all devices
- Built-in navigation and state management

### 5. **Deployment Ready**
- Easy deployment to Streamlit Cloud, Heroku, or AWS
- Automatic dependency management
- Built-in sharing capabilities

### Alternative Frameworks Considered:
- **Flask/Django**: More complex setup, requires frontend development
- **Dash**: Similar to Streamlit but more verbose
- **Tkinter**: Desktop-only, less modern interface
- **Jupyter Notebooks**: Not suitable for production apps

## 📋 Requirements

```
streamlit==1.28.1
pandas==2.1.3
numpy==1.25.2
plotly==5.17.0
mysql-connector-python==8.2.0
pymysql==1.1.0
SQLAlchemy==2.0.23
talib-binary==0.4.25
scikit-learn==1.3.2
deap==1.4.1
configparser
datetime
```

## 🛠️ Installation & Setup

### 1. Clone or Download the Project
```bash
# If you have the files, navigate to the project directory
cd your-project-directory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Configuration
Make sure your `conf/config.ini` file is properly configured:

```ini
[mysql]
url=localhost
driver=com.mysql.cj.jdbc.Driver
username=your_username
password=your_password
database=stocksdb
table=stocksinfp
earning_table=stocks_earnings
revenue_table=stocks_revenue
stock_change_tracker_table=stock_change_tracker

[stocks]
symbols=
```

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Database Schema

Your existing MySQL database should have these tables:

### `stock_change_tracker` Table
| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Stock symbol (e.g., AAPL) |
| change_in_percent | DECIMAL | Daily change percentage |
| earning_date | DATE | Next earnings date |
| current_price | DECIMAL | Current stock price |
| previous_close | DECIMAL | Previous day's closing price |
| volume | BIGINT | Trading volume |
| market_cap | BIGINT | Market capitalization |

### `stocksinfp` Table
| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Stock symbol |
| date | DATE | Trading date |
| open | DECIMAL | Opening price |
| high | DECIMAL | Day's high price |
| low | DECIMAL | Day's low price |
| close | DECIMAL | Closing price |
| volume | BIGINT | Trading volume |

## 🤖 Technical Analysis Features

### RSI (Relative Strength Index)
- Custom implementation of 14-day RSI
- Overbought (>70) and Oversold (<30) indicators
- Trend analysis and signal generation

### Neuro-Evolution Trading Agent
- Genetic algorithm-based neural network
- Uses 10 technical indicators as input features:
  - RSI, MACD, Bollinger Bands
  - Moving averages (5, 20, 50-day)
  - Volume indicators, volatility measures
- Generates BUY/SELL/HOLD signals with confidence levels

### Technical Indicators
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price volatility indicators
- **Moving Averages**: 5, 20, and 50-day MAs
- **Volume Analysis**: Volume ratio and trends

## 🎨 User Interface

### Home Page Features
- **Color-coded Performance**: Green for positive, red for negative changes
- **Earnings Highlight**: Green circle (🟢) for stocks with earnings in next 3 weeks
- **Clickable Symbols**: Easy navigation to detailed analysis
- **Responsive Layout**: Works on desktop and mobile
- **Database Status**: Real-time connection status in sidebar

### Stock Detail Page Features
- **Tabbed Interface**: Organized analysis in three distinct tabs
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Data**: Live RSI calculations and trading signals
- **Navigation**: Easy back-to-home functionality

## 🔧 Configuration

### Database Configuration
Edit `conf/config.ini` to match your database settings:

```ini
[mysql]
url=your_host
username=your_username
password=your_password
database=your_database_name
table=your_stocksinfp_table_name
stock_change_tracker_table=your_stock_change_tracker_table_name
```

### Adding Stock Symbols
Update the symbols list in `conf/config.ini`:

```ini
[stocks]
symbols=AAPL GOOGL MSFT TSLA AMZN META NVDA AMD
```

### Modifying Technical Indicators
1. Edit the `calculate_technical_indicators()` function in `stock_analysis.py`
2. Add new indicators to the feature list in `NeuroEvolutionAgent`

### Styling Changes
1. Modify the CSS in the `st.markdown()` section of `app.py`
2. Update colors, fonts, and layout as needed

## 📈 Performance Optimization

- **Connection Caching**: MySQL connections are cached for optimal performance
- **Efficient Queries**: Optimized SQL queries for fast data retrieval
- **Data Caching**: Uses Streamlit's caching for data loading functions
- **Memory Management**: Proper cleanup of database connections

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Push code to GitHub (make sure to exclude config.ini with sensitive data)
2. Use Streamlit secrets for database configuration
3. Deploy with one click

### Local Network
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify MySQL server is running
   - Check credentials in `conf/config.ini`
   - Ensure database and tables exist
   - Check network connectivity if using remote MySQL

2. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`
   - For TA-Lib issues: `pip install talib-binary`

3. **Configuration File Not Found**
   - Ensure `conf/config.ini` exists in the project directory
   - Check file permissions

4. **No Data Displayed**
   - Verify tables contain data
   - Check table names in configuration
   - Ensure column names match expected schema

5. **Performance Issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Optimize MySQL queries
   - Add database indexes for better performance

## 🔒 Security Best Practices

- Never commit `config.ini` with real credentials to version control
- Use environment variables for production deployments
- Implement proper database user permissions
- Use SSL connections for remote MySQL databases

## 📝 Future Enhancements

- [ ] Real-time data integration (Alpha Vantage, Yahoo Finance)
- [ ] Portfolio tracking and performance metrics
- [ ] Advanced charting (candlestick, volume profiles)
- [ ] Email/SMS alerts for trading signals
- [ ] Machine learning model comparison
- [ ] Options analysis and Greeks calculation
- [ ] Backtesting framework
- [ ] Multi-timeframe analysis
- [ ] User authentication and personalized dashboards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with your database setup
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Verify your database configuration
3. Ensure all dependencies are installed
4. Check the sidebar for database connection status
5. Review MySQL logs for connection issues

---

**Happy Trading! 📈**

## 🚦 Quick Start Checklist

- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Verify `conf/config.ini` is configured correctly
- [ ] Ensure MySQL server is running and accessible
- [ ] Confirm your database tables contain data
- [ ] Run the application: `streamlit run app.py`
- [ ] Check the sidebar for database connection status
- [ ] Navigate through the interface to test functionality 