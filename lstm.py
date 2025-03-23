import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import optuna

# Load and preprocess data w/ new features 
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['TimeStamp'] = (df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df['Daily Variation'] = (df['High'] - df['Low']) / df['Open'] # Volatility of the index on a specific day
    df['Index Hash'] = df['Index'].apply(hash) # Encodes identity of each index in a unqiue way 
    df['7-Day SMA'] = df['Close'].rolling(window=7).mean() # 7 Day simple moverging average (closed column)
    df['7-Day STD'] = df['Close'].rolling(window=7).std() # 7 Day standard deviation (closed column)
    df['High - Close'] = (df['High'] - df['Close']) / df['Open'] # Downward pressure of index on a given day 
    df['Low - Open'] = (df['Low'] - df['Open']) / df['Open'] # Upwards pressure of index on a given day 
    df['Cumulative Return'] = df['Close'].pct_change().cumsum() # long-term preformance of the index
    df['14-Day EMA'] = df['Close'].ewm(span=14).mean() # 14-day exponential MA (Smother & more responseive alt to SMA)
    df['Close Change'] = df['Close'].diff() # % change in the close column from pervious day
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean() # Tracks the trend and momentum of an asset
    df['Stochastic Oscillator'] = (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) # Represents position of the index relative to its recent range
    df['ATR'] = df['Daily Variation'].rolling(14).mean() # Volatility of the index over time 
    df['ADX'] = df['Close'].rolling(14).mean() #  Strength & direction of the trend of the index
    df['DMI'] = df['Close'].rolling(14).mean() #  Positive and negative movements of the index
    
    df.dropna(inplace=True)
    return df

# Prepare training data
def prepare_data(df, target_col='Close'):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['Date', 'Index']))
    X, y = [], []
    for i in range(30, len(scaled_data)):
        X.append(scaled_data[i-30:i])
        y.append(scaled_data[i, df.columns.get_loc(target_col)-2])
    return np.array(X), np.array(y), scaler

# Build LSTM model
def lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and evaluate model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = lstm((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return model, history, metrics, y_test, y_pred


# LSTM w/ Optuna optimization
def optunaLSTM(trial, X_train, X_test, y_train, y_test):
    units_1 = trial.suggest_int('units_1', 32, 128)
    units_2 = trial.suggest_int('units_2', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 20, 100)
    
    model = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Main function
def main():
    df = load_data('Stock Data.csv')
    X, y, scaler = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate standard LSTM
    model_lstm, history_lstm, metrics_lstm, _, y_pred_lstm = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Optimize LSTM with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optunaLSTM(trial, X_train, X_test, y_train, y_test), n_trials=10)
    best_params = study.best_params
    print("Best Parameters:", best_params)
    
    # Train optimized model
    model_opt = Sequential([
        LSTM(best_params['units_1'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(best_params['dropout_rate']),
        LSTM(best_params['units_2']),
        Dropout(best_params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    model_opt.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])
    history_opt = model_opt.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                              validation_data=(X_test, y_test), verbose=0)
    y_pred_opt = model_opt.predict(X_test)
    
    # Calculate metrics for optimized model
    metrics_opt = {
        "MAE": mean_absolute_error(y_test, y_pred_opt),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred_opt),
        "MSE": mean_squared_error(y_test, y_pred_opt),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_opt))
    }
    
    # Print metrics
    print("\nStandard LSTM Metrics:", metrics_lstm)
    print("Optimized LSTM Metrics:", metrics_opt)
    
    # Plot validation loss comparison
    plt.figure(figsize=(10,5))
    plt.plot(history_lstm.history['val_loss'], label='Non-Optimized LSTM Validation Loss')
    plt.plot(history_opt.history['val_loss'], label='Optimized LSTM Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
  
if __name__ == "__main__":
    main()
