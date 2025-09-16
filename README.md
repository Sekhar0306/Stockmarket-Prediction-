This project provides an interactive interface for time-series forecasting. Users can upload a CSV file containing historical stock data, configure the LSTM model parameters, and visualize both the model's performance on historical data and its predictions for the future. The application is designed to be flexible, even allowing for custom date formats to handle a wide variety of datasets.

Interactive Web Interface: A clean and modern UI built with Streamlit.

Custom Data Upload: Upload your own stock data in CSV format.

Flexible Data Handling: Select your date and value columns, and specify custom date formats if needed.

Tunable LSTM Model: Adjust key model parameters like the look-back window, epochs, batch size, and train/test split.

Rich Visualizations: Interactive charts powered by Plotly to display historical prices, model fit, and future forecasts.

Performance Metrics: A dashboard shows the last known price and the predicted price for the next day, including the percentage change.


