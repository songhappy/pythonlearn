import xgboost as xgb
import preprocess_utils
from sklearn.metrics import mean_squared_log_error
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Activation, multiply, Input, LSTM
from tensorflow.keras.optimizers import Adam

prediction_step= 1
testdatasize = 100
unroll_length = 15
label_vars = ['spx_close', 'ixic_close', 'dji_close', 'iwm_close',
              'soxx_close']
yahoo_etf_input = "/Users/guoqiong/life/invest/stock/yahoo_data/etf.txt"

def attention_one(x_train, y_train, x_test, y_test, features):
    shape = features.shape
    features = features.reshape(-1, shape[0], shape[1])
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)

    cnn = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # padding = 'same'

    # Define the attention mechanism layer
    def attention_3d_block(inputs):
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(
            hidden_states)
        score_activation = Activation('tanh')(score_first_part)
        attention_weights = Dense(1, name='attention_weight_vec')(score_activation)
        attention_weights = Activation('softmax')(attention_weights)
        context_vector = multiply([hidden_states, attention_weights])
        return context_vector

    # Build the model

    lstm_out = LSTM(64, return_sequences=True)(cnn)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    output = Dense(1)(attention_flatten)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=False)

    # Evaluate the model
    mse = model.evaluate(x_test, y_test)
    # print('Mean Squared Error:', mse)

    # Make predictions
    predictions = model.predict(features)
    return predictions

def cnn_one(x_train, y_train, x_test, y_test, features):

    shape = features.shape
    features = features.reshape(-1, shape[0], shape[1])

    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=(5), padding='Same',
                     activation='relu', input_shape=shape))
    model.add(Conv1D(filters=32, kernel_size=(5), padding='Same',
                     activation='relu'))
    model.add(MaxPool1D(pool_size=(5)))

    model.add(Conv1D(filters=64, kernel_size=(3), padding='Same',
                     activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=(3), padding='Same',
                     activation='relu'))
    model.add(MaxPool1D(pool_size=(3), strides=(2)))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    model.fit(x_train, y_train, epochs=10, verbose=False)
    predictions = model(features)
    return predictions.numpy()


def xgb_one(x_train, y_train, x_test, y_test, features):

    x_train = x_train.reshape(x_train.shape[0], -1 )
    x_test = x_test.reshape(x_test.shape[0], -1)
    features = features.reshape(-1, feature_size * unroll_length)
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=20)
    model_xgb.fit(x_train, y_train)
    y_hat = model_xgb.predict(x_test)

    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_hat))
    predictions = model_xgb.predict(features)

    return predictions, RMSLE


if __name__ == '__main__':
    df_in = preprocess_utils.get_df(yahoo_etf_input)
    column_names = df_in.columns.values
    print(column_names)
    df_scaled, x_scaler = preprocess_utils.scale_x(df_in)
    x_train, x_test = preprocess_utils.generate_x(df_scaled, testdatasize, unroll_length, prediction_step)

    report=[]
    for label_var in label_vars:
        # y_scaled, y_scaler = preprocess_utils.scale_y(df_in[[label_var]], label_var)
        # y_train, y_test = preprocess_utils.generate_y(y_scaled, label_var, len(x_train), len(x_test),
        #                                               testdatasize, unroll_length, prediction_step)

        df_scaled, x_scaler, y_scaler = preprocess_utils.scale(df_in, label_var)
        x_train, y_train, x_test, y_test = preprocess_utils.generate_train_test(df_scaled, label_var, testdatasize, unroll_length, prediction_step)

        feature_size = len(df_scaled.columns.values)
        features = df_scaled[-unroll_length:]
        features = features.to_numpy()

        predictions_raw1, RMSLE = xgb_one(x_train, y_train, x_test, y_test, features)
        predictions_raw2 = cnn_one(x_train, y_train,x_test, y_test, features)
        predictions_raw3 = attention_one(x_train, y_train,x_test, y_test, features)

        predictions1 = y_scaler.inverse_transform(predictions_raw1.reshape(-1, 1))
        predictions2 = y_scaler.inverse_transform(predictions_raw2.reshape(-1, 1))
        predictions3 = y_scaler.inverse_transform(predictions_raw3.reshape(-1, 1))

        today = df_in[label_var][len(df_in) - 1]
        # print([predictions, today])
        direction1 = (predictions1[0, 0] - today)/today
        direction2 = (predictions2[0, 0] - today)/today
        direction3 = (predictions3[0, 0] - today)/today

        report.append([label_var, direction1, direction2, direction3])

    for item in report:
        print(item)

