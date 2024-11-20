# Time Series Forecasting Project

## Project Overview

This project demonstrates the use of time series forecasting to predict future values based on historical data. Specifically, it applies the ARIMA (AutoRegressive Integrated Moving Average) model to forecast future temperature values using a publicly available dataset. The project involves data cleaning, exploration, modeling, evaluation, and visualization of the results.

### Dataset

The dataset used for this project is the **Daily Minimum Temperatures in Melbourne** dataset, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Daily+Minimum+Temperatures). The dataset consists of daily minimum temperatures recorded over a period of several years. The main goal of this project is to forecast future temperature values.

### Steps Involved

1. **Data Loading**: Load and preprocess the dataset.
2. **Data Visualization**: Visualize the time series data to understand trends and patterns.
3. **Modeling**: Apply the ARIMA model for forecasting.
4. **Evaluation**: Evaluate the model using Mean Squared Error (MSE).
5. **Visualization**: Plot the forecast against actual values.

## Prerequisites

Before running the project, you need to install the following libraries:

- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn

You can install these dependencies using the following command:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

## Project Setup

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/time-series-forecasting.git
   ```
2. Change into the project directory:
   ```bash
   cd time-series-forecasting
   ```

## File Structure

```bash
time-series-forecasting/
├── data/
│   └── daily-min-temperatures.csv  # Dataset file
├── notebook/
│   └── forecasting.ipynb  # Jupyter notebook for model implementation
├── output/
│   └── forecast_plot.png  # Plot output after running the script
└── README.md  # This file
```

## How to Run the Project

### Step 1: Load and Preprocess Data

The first step involves loading the dataset into a pandas DataFrame and preprocessing it. Missing values are dropped, and the date is parsed to ensure proper handling.

### Step 2: Data Visualization

Visualize the historical temperature data to explore its trends and seasonality.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/daily-min-temperatures.csv', parse_dates=['Date'], index_col='Date')

# Plot the data
data.plot(figsize=(12, 6))
plt.title('Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()
```

### Step 3: Build ARIMA Model

After data preprocessing and exploration, the ARIMA model is trained on the data, and a forecast is generated.

```python
from statsmodels.tsa.arima.model import ARIMA

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test)).values
```

### Step 4: Evaluate the Model

The performance of the model is evaluated using the Mean Squared Error (MSE) metric:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse}')
```

### Step 5: Visualize the Forecast

The results of the forecast are plotted alongside the actual values:

```python
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

## Output

The output of this project includes:
- **Forecast Plot**: A visualization comparing actual vs forecasted temperature values.
- **Model Evaluation**: MSE to evaluate the performance of the model.

## Conclusion

This project demonstrates how time series forecasting can be applied to predict future values using historical data. By leveraging ARIMA, we can make predictions and assess model accuracy. This approach can be extended to other datasets and more advanced forecasting techniques like SARIMA or LSTM.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
