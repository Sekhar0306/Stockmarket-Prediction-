import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

pio.templates.default = "plotly_dark"

tf.config.set_visible_devices([], 'GPU')

st.set_page_config(
    page_title="QuantumLeap | Stock Forecaster",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        background-color: #0E1117;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stMetric > div:nth-child(2) > div:nth-child(1) {
        font-size: 2em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data(data_source, date_col, value_col, date_format):
    """Loads, cleans, and prepares the stock data from a CSV."""
    try:
        df = pd.read_csv(data_source)
        df = df[[date_col, value_col]].copy()
        
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
            
        df.sort_values(by=date_col, inplace=True)
        df.set_index(date_col, inplace=True)
        df.rename(columns={value_col: 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error processing data: {e}. Please check your column selections and date format string.")
        return None

def create_dataset(dataset, look_back=1):
    """Creates input-output pairs for the LSTM model."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

st.title("💸 QuantumLeap Stock Forecaster")
st.markdown("Train a custom LSTM model to predict future stock prices.")

st.sidebar.header("⚙️ Model Configuration")
st.sidebar.subheader("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your historical data (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file using the sidebar to begin.")

if uploaded_file is not None:
    data_source_name = uploaded_file.name.split('.')[0]

    temp_df_cols = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
    uploaded_file.seek(0)

    st.sidebar.subheader("2. Column Selection")
    date_col = st.sidebar.selectbox("Date Column", temp_df_cols, index=0)
    
    date_format_string = st.sidebar.text_input(
        "Date Format String (optional)",
        help="Example: %Y-%m-%d. Leave empty to auto-detect. [See format codes](https://strftime.org/)"
    )

    value_col = st.sidebar.selectbox("Value Column to Predict", temp_df_cols, index=min(4, len(temp_df_cols)-1))
    
    data_df = load_and_preprocess_data(uploaded_file, date_col, value_col, date_format_string)

    if data_df is not None:
        with st.expander(f"Raw Data Preview for {data_source_name}", expanded=False):
            st.dataframe(data_df.tail())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_df.index, y=data_df['Price'], mode='lines', name='Actual Price', line=dict(color='#00BFFF')))
        fig.update_layout(title=f'Historical Price Chart: {data_source_name}', xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)

        st.sidebar.markdown("---")
        st.sidebar.subheader("3. LSTM Parameters")
        look_back = st.sidebar.slider("Look-back Window (days)", 10, 120, 60, 5)
        epochs = st.sidebar.slider("Training Epochs", 10, 200, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 1, 64, 32)
        train_test_ratio = st.sidebar.slider("Training Data Ratio", 0.1, 0.9, 0.8, 0.05)
        
        st.sidebar.subheader("4. Prediction Horizon")
        future_prediction_days = st.sidebar.number_input("Days to Predict Ahead", 1, 365, 30)
        st.sidebar.markdown("---")

        if st.sidebar.button("🚀 Generate Forecast", use_container_width=True, type="primary"):
            with st.status("🚀 **Training LSTM Model...**", expanded=True) as status:
                st.write("1/5: Scaling and preparing data...")
                dataset = data_df['Price'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_dataset = scaler.fit_transform(dataset)
                
                train_size = int(len(scaled_dataset) * train_test_ratio)
                train, test = scaled_dataset[0:train_size, :], scaled_dataset[train_size:len(scaled_dataset), :]
                trainX, trainY = create_dataset(train, look_back)
                testX, testY = create_dataset(test, look_back)
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
                testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

                st.write("2/5: Building LSTM architecture...")
                model = Sequential([
                    Input(shape=(look_back, 1)),
                    LSTM(50, return_sequences=True),
                    LSTM(50, return_sequences=False),
                    Dense(25),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')

                st.write(f"3/5: Training for {epochs} epochs...")
                model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=0)
                
                st.write("4/5: Making validation predictions...")
                train_predict = scaler.inverse_transform(model.predict(trainX))
                test_predict = scaler.inverse_transform(model.predict(testX))
                
                st.write("5/5: Forecasting future values...")
                temp_input = list(scaled_dataset[-look_back:].flatten())
                future_predictions_list = []
                for _ in range(future_prediction_days):
                    x_input = np.array(temp_input[-look_back:]).reshape((1, look_back, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.append(yhat[0, 0])
                    future_predictions_list.append(yhat[0, 0])
                
                future_predictions = scaler.inverse_transform(np.array(future_predictions_list).reshape(-1, 1))
                status.update(label="✅ **Forecast Complete!**", state="complete", expanded=False)

            st.header(f"Forecast Analysis: {data_source_name}")
            
            last_price = data_df['Price'].iloc[-1]
            next_day_prediction = future_predictions[0, 0]
            change = next_day_prediction - last_price
            percent_change = (change / last_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Last Recorded Price", f"${last_price:,.2f}")
            col2.metric("Predicted Next Day Price", f"${next_day_prediction:,.2f}", f"{change:,.2f} ({percent_change:.2f}%)")
            col3.metric("Prediction Horizon", f"{future_prediction_days} Days")
            
            st.markdown("---")

            tab1, tab2 = st.tabs(["Future Forecast", "Model Performance"])

            with tab1:
                st.subheader(f"Price Forecast for the Next {future_prediction_days} Days")
                last_date = data_df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_prediction_days)
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=data_df.index[-200:], y=data_df['Price'][-200:], mode='lines', name='Historical Price', line=dict(color='#00BFFF')))
                fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Forecasted Price', line=dict(color='#FFD700', dash='dash')))
                fig_future.update_layout(title_text='Future Price Forecast', yaxis_title='Price (USD)')
                st.plotly_chart(fig_future, use_container_width=True)

            with tab2:
                st.subheader("Model Fit on Historical Data")
                fig_performance = go.Figure()
                fig_performance.add_trace(go.Scatter(x=data_df.index, y=data_df['Price'], mode='lines', name='Actual Values', line=dict(color='#00BFFF', width=2)))
                train_predict_plot = np.empty_like(dataset); train_predict_plot[:, :] = np.nan
                train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
                fig_performance.add_trace(go.Scatter(x=data_df.index, y=train_predict_plot.flatten(), mode='lines', name='Training Prediction', line=dict(color='orange', dash='dot', width=1.5)))
                test_predict_plot = np.empty_like(dataset); test_predict_plot[:, :] = np.nan
                test_predict_plot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict
                fig_performance.add_trace(go.Scatter(x=data_df.index, y=test_predict_plot.flatten(), mode='lines', name='Validation Prediction', line=dict(color='lightgreen', dash='dot', width=1.5)))
                fig_performance.update_layout(title_text='Model Performance on Training & Validation Sets', yaxis_title='Price (USD)')
                st.plotly_chart(fig_performance, use_container_width=True)
            
            st.success("Analysis complete. Adjust parameters in the sidebar to run a new forecast.")