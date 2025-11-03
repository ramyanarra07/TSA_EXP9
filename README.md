# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 3-11-2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- Load Dataset ---
file_path = '/content/9. Sales-Data-Analysis.csv'
data = pd.read_csv(file_path)

# **********************************************************************
# ADAPTATION for '9. Sales-Data-Analysis.csv':
# 1. Convert 'Date' to datetime.
# 2. Calculate Daily Revenue.
# 3. Resample to create a daily time series.
# **********************************************************************

# 1. Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# 2. Calculate Revenue (Price * Quantity)
data['Revenue'] = data['Price'] * data['Quantity']

# 3. Resample to Daily Total Revenue
daily_revenue_ts = data.groupby('Date')['Revenue'].sum().to_frame()
daily_revenue_ts.rename(columns={'Revenue': 'Daily_Revenue'}, inplace=True)
daily_revenue_ts.sort_index(inplace=True)

# Select 'Daily_Revenue' as the target variable
target_variable = 'Daily_Revenue'
data_ts = daily_revenue_ts.copy()

print(f"Set target column: {target_variable}")
print("First 5 rows of Daily Revenue time series:")
print(data_ts.head())
print("Data shape:", data_ts.shape)

# --- Define ARIMA Modeling Function ---
def arima_model(data, target_variable, order=(1,1,1)):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Print Model Summary (useful for checking parameters)
    print("\nARIMA Model Summary:")
    print(fitted_model.summary())

    # Forecast
    # Use INTEGER INDICES for forecasting (start is the first index after train_data)
    forecast = fitted_model.predict(start=len(train_data), end=len(data) - 1)
    
    # Assign the correct datetime index to the forecast results for plotting
    forecast.index = test_data.index
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(forecast.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title(f'ARIMA ({order}) Forecasting for Daily Revenue (Sales Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return fitted_model, forecast

# --- Run ARIMA Model ---
# Running with the requested order=(5,1,0) on Daily Revenue
model, forecast = arima_model(data_ts, target_variable, order=(5,1,0))
```

### OUTPUT:

<img width="678" height="691" alt="image" src="https://github.com/user-attachments/assets/ae2a83ab-b807-4b1d-ad25-021013914f05" />

<img width="953" height="572" alt="image" src="https://github.com/user-attachments/assets/387dd79e-d349-45a0-a323-67a8625eba51" />



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
