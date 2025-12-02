# === IMPORT LIBRARIES ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === STEP 1: FETCH STOCK DATA ===
def load_stock_data(ticker='AAPL', start='2015-01-01'):
    end = pd.Timestamp.today().strftime('%Y-%m-%d')  # today's date
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = load_stock_data()

# === STEP 2: PREPROCESSING ===
def preprocess_data(df, sequence_len=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_len, len(scaled_data)):
        X.append(scaled_data[i-sequence_len:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, scaled_data

sequence_len = 60
X, y, scaler, scaled_data = preprocess_data(df, sequence_len)

# === STEP 3: SPLIT INTO TRAIN & TEST ===
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# === STEP 4: BUILD LSTM MODEL ===
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm_model((X_train.shape[1], 1))

# === STEP 5: TRAIN THE MODEL ===
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# === STEP 6: MAKE PREDICTIONS ON TEST DATA ===
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# === STEP 7: VISUALIZE RESULTS ===
def plot_predictions(real, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_predictions(real_prices, predicted_prices)

# === STEP 8: PREDICT TODAY'S PRICE ===
def predict_today_price(model, scaled_data, scaler, sequence_len=60):
    # Take last 60 days
    last_sequence = scaled_data[-sequence_len:]
    X_today = np.reshape(last_sequence, (1, sequence_len, 1))
    
    # Predict
    predicted_today = model.predict(X_today)
    predicted_today = scaler.inverse_transform(predicted_today)
    return predicted_today[0][0]

today_price_pred = predict_today_price(model, scaled_data, scaler, sequence_len)
print("Predicted Apple closing price today:", today_price_pred)

# === STEP 9: FETCH ACTUAL TODAY'S PRICE (for comparison) ===
today_actual = yf.download('AAPL', start=pd.Timestamp.today().strftime('%Y-%m-%d'),
                           end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))['Close']
print("Actual Apple closing price today:", today_actual.values if not today_actual.empty else "Market not closed yet")
