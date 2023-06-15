import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

def get_df(file_name):
    file_name = "/Users/guoqiong/life/invest/stock/yahoo_data/etf.txt"
    df_org = pd.read_csv(file_name)
    print(df_org[:5])
    # print(df_org[-5:])

    df_org = df_org.iloc[:, :-1].iloc[:, 1:]
    # print(df_org[:5])

    def get_names(df):
        new_names = []
        for i in range(len(df.loc[0])):
            str1 = str(df.iloc[0, i]).replace("^","").lower()
            str2 = str(df.columns.values[i]).split(".")[0].replace(" ", "_").replace(" ", "_").lower()
            new_name = "_".join([str1,str2])
            new_names.append(new_name)
        old_names = df.columns.values
        names_dict = dict(zip(old_names, new_names))
        return names_dict, new_names

    names_dict, new_names = get_names(df_org)
    df_org.rename(columns=names_dict, inplace=True)
    # print(df_org.columns.values)

    # print(new_names)
    df_org = df_org[2:].reset_index(drop=True)
    df = df_org.astype(float)
    return df


def scale_x(df):
    column_names = df.columns.values
    x_scaler = preprocessing.MinMaxScaler()
    df_scaled = x_scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=column_names)
    return df_scaled, x_scaler


def scale_y(df, label_var):
    y_scaler = preprocessing.MinMaxScaler()
    y = y_scaler.fit_transform(df[[label_var]])
    y_scaled = pd.DataFrame(y, columns=[label_var])
    return y_scaled, y_scaler


def scale(df, label_var):
    column_names = df.columns.values
    y_scaler = preprocessing.MinMaxScaler()
    y = y_scaler.fit_transform(df[[label_var]])
    y_scaled = pd.DataFrame(y, columns=[label_var])

    x_scaler = preprocessing.MinMaxScaler()
    df_scaled = x_scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=column_names)
    df_scaled[[label_var]] = y_scaled[[label_var]]
    return df_scaled, x_scaler, y_scaler


def unroll(data, sequence_length):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)


def generate_x(df,
               testdatasize=200,
               unroll_length=30,
               prediction_step=1):
    testdatacut = testdatasize + unroll_length + prediction_step
    data = df.to_numpy()
    unrolled = unroll(data, unroll_length)
    print("shape")
    print(data.shape)
    print(unrolled.shape)
    x_train = unrolled[:-testdatasize]
    x_test = unrolled[-testdatasize:]
    return x_train, x_test


def generate_y(df, label_var, len_train, len_test,
               testdatasize=200,
               unroll_length=30,
               prediction_step=1):
    y_train = df[unroll_length:len_train + unroll_length][label_var].to_numpy()
    y_test = df[-testdatasize:][label_var].to_numpy()
    return y_train, y_test


def generate_train_test(df, label_var,
                        testdatasize=200,
                        unroll_length=30,
                        prediction_step=1):

    # testdatacut = testdatasize + unroll_length + prediction_time
    # x_train = df[0:-prediction_time-testdatacut].to_numpy()
    # y_train = df[prediction_time:-testdatacut][label_var].to_numpy()
    # x_test = df[-testdatacut:-prediction_time].to_numpy()
    # y_test = df[prediction_time-testdatacut:][label_var].to_numpy()

    testdatacut = testdatasize + unroll_length + prediction_step
    x_train = df[:].to_numpy()
    y_train = df[:][label_var].to_numpy()
    x_test = df[-testdatacut:-prediction_step].to_numpy()
    y_test = df[prediction_step-testdatacut:][label_var].to_numpy()

    def unroll(data, sequence_length):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)

    # adapt the datasets for the sequence data shape
    x_train = unroll(x_train, unroll_length)
    x_test = unroll(x_test, unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test = y_test[-x_test.shape[0]:]
    return x_train, y_train, x_test, y_test