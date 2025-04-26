import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="Electricity Consumption Forecaster", layout="wide")

st.title("⚡ Electricity Consumption Forecaster")
st.write("Upload your electricity consumption data and predict future consumption values")

def train_arima_model(df, order=(2, 1, 2)):
    """
    Trains an ARIMA model on the dataset and returns the fitted model.
    """
    # Ensure we have hourly frequency
    df = df.asfreq('H')
    
    # Fill missing data if any
    df['consumption'].fillna(method='ffill', inplace=True)
    
    # Train ARIMA model on the entire dataset
    try:
        model = ARIMA(df['consumption'], order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None

def forecast_for_datetime(df, model_fit, forecast_date):
    """
    Forecast power consumption for a specific date/time
    """
    # Last known timestamp in the dataset
    last_timestamp = df.index[-1]
    
    # If user_dt <= last_timestamp, we can't forecast "backwards"
    if forecast_date <= last_timestamp.to_pydatetime():
        st.warning(f"⚠️ Requested date ({forecast_date}) is not in the future compared to the last data point ({last_timestamp}). Returning historical value instead.")
        # Find the closest available value in historical data
        closest_idx = df.index.get_indexer([forecast_date], method='nearest')[0]
        return df['consumption'].iloc[closest_idx], True
    
    # Calculate how many hours ahead we need to forecast
    time_diff = forecast_date - last_timestamp.to_pydatetime()
    hours_ahead = int(time_diff.total_seconds() // 3600) + 1  # +1 to ensure we include the target hour
    
    # Forecast that many steps ahead
    forecast_values = model_fit.forecast(steps=hours_ahead)
    
    # The last forecasted value is for the user_dt
    predicted_value = forecast_values.iloc[-1]
    return predicted_value, False

def calculate_cost(consumption, rate_per_100_units):
    """Calculate the cost based on consumption and rate per 100 units"""
    return (consumption / 100) * rate_per_100_units

def generate_download_link(df, filename="forecast_results.csv"):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def plot_forecast(df, forecast_df, forecast_date, prediction):
    """Create a plot showing historical data and forecast"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['consumption'],
        mode='lines',
        name='Historical Consumption',
        line=dict(color='blue')
    ))
    
    # Plot forecasted data if available
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecasted Values',
            line=dict(color='red', dash='dash')
        ))
    
    # Highlight the specific forecast point
    fig.add_trace(go.Scatter(
        x=[forecast_date],
        y=[prediction],
        mode='markers',
        name='Forecasted Point',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title='Electricity Consumption: Historical and Forecast',
        xaxis_title='Date',
        yaxis_title='Consumption (kWh)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Sidebar for configuration
with st.sidebar:
    st.header("Model Configuration")
    
    # ARIMA order parameters
    st.subheader("ARIMA Parameters")
    p = st.slider("p (AR order)", 0, 5, 2)
    d = st.slider("d (Differencing)", 0, 2, 1)
    q = st.slider("q (MA order)", 0, 5, 2)
    
    # Forecast horizon
    st.subheader("Forecast Horizon")
    forecast_type = st.radio("Forecast type:", ["Single date", "Range of dates"])
    
    # Electricity rate selection
    st.subheader("Electricity Rate")
    rate_options = {
        "₹5 per 100 units": 5,
        "₹7 per 100 units": 7,
        "₹10 per 100 units": 10
    }
    selected_rate_option = st.selectbox(
        "Select electricity rate:",
        options=list(rate_options.keys()),
        index=0
    )
    # Extract the numeric rate value
    selected_rate = rate_options[selected_rate_option]
    
    # Advanced options
    st.subheader("Advanced Options")
    confidence_interval = st.checkbox("Show confidence intervals", value=False)
    
    # About section
    st.subheader("About")
    st.markdown("""
    This app uses ARIMA models to forecast electricity consumption based on historical data.
    
    Upload a CSV file with columns:
    - timestamp (date/time)
    - consumption (numeric values)
    """)

# Main content
uploaded_file = st.file_uploader("Upload your electricity consumption CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'], index_col='timestamp')
        
        # Check if required columns exist
        if 'consumption' not in df.columns:
            st.error("CSV must contain a 'consumption' column!")
            st.stop()
            
        # Display data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Date range**: {df.index.min().date()} to {df.index.max().date()}")
            st.write(f"**Number of records**: {len(df)}")
        with col2:
            st.write(f"**Average consumption**: {df['consumption'].mean():.2f} kWh")
            st.write(f"**Data frequency**: {pd.infer_freq(df.index) or 'Unable to detect'}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Plot historical data
        st.subheader("Historical Consumption")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['consumption'], mode='lines'))
        fig.update_layout(title='Historical Electricity Consumption',
                         xaxis_title='Date',
                         yaxis_title='Consumption (kWh)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Train model
        with st.spinner("Training ARIMA model..."):
            arima_order = (p, d, q)
            model_fit = train_arima_model(df, order=arima_order)
            
        if model_fit is not None:
            st.success("Model training complete!")
            
            # Display model summary - FIXED: Removed problematic code that was causing the error
            with st.expander("Model Summary"):
                # Display a simpler summary that doesn't rely on writerow
                st.text(str(model_fit.summary()))
            
            # Forecast section
            st.subheader("Forecast Electricity Consumption")
            
            if forecast_type == "Single date":
                # Single date forecast
                col1, col2 = st.columns(2)
                with col1:
                    forecast_date = st.date_input("Select date for forecast", 
                                              value=df.index.max().date() + datetime.timedelta(days=1))
                with col2:
                    forecast_time = st.time_input("Select time", 
                                              value=datetime.time(hour=12, minute=0))
            
                forecast_datetime = datetime.datetime.combine(forecast_date, forecast_time)
                
                if st.button("Predict Consumption"):
                    with st.spinner("Generating forecast..."):
                        prediction, is_historical = forecast_for_datetime(df, model_fit, forecast_datetime)
                        
                        # Calculate cost based on selected rate
                        cost = calculate_cost(prediction, selected_rate)
                        
                        # Create a result container with styling
                        st.markdown("### Forecast Result")
                        result_container = st.container()
                        
                        with result_container:
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.markdown(f"**Date and Time:**  \n{forecast_datetime.strftime('%Y-%m-%d %H:%M')}")
                            with col2:
                                st.markdown(f"**Predicted Consumption:**  \n{prediction:.3f} kWh")
                            with col3:
                                st.markdown(f"**Estimated Cost:**  \n₹{cost:.2f} ({selected_rate_option})")
                                
                            if is_historical:
                                st.warning("⚠️ This is a historical value, not a forecast")
                            
                            # Create a forecast dataframe for visualization
                            if not is_historical:
                                # Generate forecast series from last timestamp to forecast_datetime
                                last_timestamp = df.index[-1]
                                time_diff = forecast_datetime - last_timestamp.to_pydatetime()
                                hours_ahead = int(time_diff.total_seconds() // 3600) + 1
                                
                                forecast_values = model_fit.forecast(steps=hours_ahead)
                                forecast_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), 
                                                            periods=hours_ahead, freq='H')
                                forecast_df = pd.DataFrame({'forecast': forecast_values}, index=forecast_dates)
                            else:
                                forecast_df = None
                                
                            # Plot the forecast
                            fig = plot_forecast(df, forecast_df, forecast_datetime, prediction)
                            st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Range of dates forecast
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start date", 
                                          value=df.index.max().date() + datetime.timedelta(days=1))
                with col2:
                    end_date = st.date_input("End date", 
                                        value=df.index.max().date() + datetime.timedelta(days=7))
                
                forecast_freq = st.selectbox("Forecast frequency", ["Hourly", "Daily", "Weekly"], index=0)
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        # Determine frequency
                        freq_map = {"Hourly": "H", "Daily": "D", "Weekly": "W"}
                        freq = freq_map[forecast_freq]
                        
                        # Create date range
                        last_timestamp = df.index[-1]
                        start_datetime = datetime.datetime.combine(start_date, datetime.time(0, 0))
                        end_datetime = datetime.datetime.combine(end_date, datetime.time(23, 0))
                        
                        # Check if dates are in future
                        if end_datetime <= last_timestamp.to_pydatetime():
                            st.error("Selected date range must be in the future compared to your data.")
                        else:
                            # Adjust start_datetime if needed
                            if start_datetime <= last_timestamp.to_pydatetime():
                                start_datetime = last_timestamp.to_pydatetime() + datetime.timedelta(hours=1)
                                st.warning(f"Start date adjusted to {start_datetime} to ensure it's after the last data point.")
                            
                            # Generate forecast dates
                            forecast_dates = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)
                            
                            # Calculate hours ahead
                            time_diff = end_datetime - last_timestamp.to_pydatetime()
                            hours_ahead = int(time_diff.total_seconds() // 3600) + 1
                            
                            # Generate forecast
                            forecast_values = model_fit.forecast(steps=hours_ahead)
                            
                            # Match forecast values to dates
                            all_hours = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), 
                                                   periods=hours_ahead, freq='H')
                            all_forecasts = pd.DataFrame({'forecast': forecast_values}, index=all_hours)
                            
                            # Filter to requested dates only
                            forecast_df = all_forecasts[all_forecasts.index.isin(forecast_dates)]
                            
                            # Add cost calculation to forecast dataframe
                            forecast_df['cost'] = forecast_df['forecast'].apply(lambda x: calculate_cost(x, selected_rate))
                            
                            # Display forecast results
                            st.markdown("### Forecast Results")
                            st.markdown(f"**Selected Rate:** {selected_rate_option}")
                            st.dataframe(forecast_df)
                            
                            # Display total cost
                            total_consumption = forecast_df['forecast'].sum()
                            total_cost = forecast_df['cost'].sum()
                            
                            st.markdown(f"**Total Predicted Consumption:** {total_consumption:.2f} kWh")
                            st.markdown(f"**Total Estimated Cost:** ₹{total_cost:.2f}")
                            
                            # Download link
                            st.markdown(generate_download_link(forecast_df), unsafe_allow_html=True)
                            
                            # Plot the forecast
                            fig = go.Figure()
                            
                            # Plot historical data
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df['consumption'],
                                mode='lines',
                                name='Historical Consumption',
                                line=dict(color='blue')
                            ))
                            
                            # Plot forecasted data
                            fig.add_trace(go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['forecast'],
                                mode='lines',
                                name='Forecasted Values',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title='Electricity Consumption Forecast',
                                xaxis_title='Date',
                                yaxis_title='Consumption (kWh)',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create a second plot for costs
                            cost_fig = go.Figure()
                            cost_fig.add_trace(go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['cost'],
                                mode='lines',
                                name='Estimated Cost',
                                line=dict(color='green')
                            ))
                            
                            cost_fig.update_layout(
                                title=f'Electricity Cost Forecast ({selected_rate_option})',
                                xaxis_title='Date',
                                yaxis_title='Cost (₹)',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(cost_fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    # Display instructions when no file is uploaded
    st.info("Please upload a CSV file to get started.")
    
    # Example of expected data format
    st.subheader("Expected CSV Format")
    example_data = {
        'timestamp': ['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00'],
        'consumption': [3.24, 2.98, 2.56]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
  

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For better results, provide at least one year of historical data with consistent intervals.")
