from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

app = Flask(__name__)

# Set random seed
np.random.seed(42)
tensorflow.random.set_seed(42)

@app.route('/predict_car_purchase', methods=['POST'])
def predict_car_purchase():
    json_data = request.get_json()
    input_data = [[json_data['Gender'], json_data['Age'], json_data['Annual Salary'], json_data['Credit Card Debt'], json_data['Net Worth']]]
    
    car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = car_df['Car Purchase Amount']
    y = y.values.reshape(-1,1)
    y_scaled = scaler.fit_transform(y)
    X_Testing = np.array(input_data)
    model = Sequential()
    model.add(Dense(25, input_dim=5, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    epochs_hist = model.fit(X_scaled, y_scaled, epochs=50, batch_size=25,  verbose=1, validation_split=0.2)
    y_predict = model.predict(X_Testing)
    
    return jsonify({'Expected Purchase Amount': str(y_predict[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
