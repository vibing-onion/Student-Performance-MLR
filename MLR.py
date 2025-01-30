import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

def wrangle(relativefilepath):
    data = pd.read_csv(relativefilepath)
    print(data.corr())
    print(data.describe())
    data = data.drop(['Sleep Hours','Sample Question Papers Practiced','Extracurricular Activities'], axis = 1)
    return data

def make_model():
    model = make_pipeline(
        OneHotEncoder(),
        LinearRegression()
    )
    return model

def model_eval(model, x_train, x_test, y_train, y_test):
    model.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
    y_pred = model.predict(np.array(x_test).reshape(-1, 1))
    
    return{'x': x_test, 'y': y_test, 'Predictions': y_pred, 'MSE': mean_squared_error(y_test, y_pred), 'R2': r2_score(y_test, y_pred)}
    
def linear_regression(data):
    result = {}
    for i in range(len(data.columns)-1):
        print(data.columns[i])
        x_train, x_test, y_train, y_test = train_test_split(data[data.columns[i]], data["Performance Index"], test_size = 0.2, random_state = 42)
        model = make_model()
        result[data.columns[i]] = model_eval(model, x_train.values, x_test.values, y_train.values, y_test.values)
    return result

def linear_regression_summary(result):
    metric = {
        col: {key: result[col][key] for key in ['MSE', 'R2']} for col in result.keys()
    }
    metric = pd.DataFrame(metric)
    print(metric)
    
def multiple_linear_regression(data):
    result = {}
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,0:-1], data["Performance Index"], test_size = 0.2, random_state = 42)
    model = make_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = pd.DataFrame({'MSE': mean_squared_error(y_test, y_pred), 'R2': r2_score(y_test, y_pred)}, index = ['Multiple Linear Regression'])
    scipy_model = sm.OLS(y_train, x_train).fit()
    print(scipy_model.summary())
    print(result)
    return {'y': y_test, 'Predictions': y_pred}
    

def plot(data, lr_result, mlr_result, axes):
    data_col = data.columns
    for i in range(len(data_col)-1):
        axes[0][i].scatter(y = data["Performance Index"], x = data[data_col[i]], marker = "+")
        axes[1][i].scatter(lr_result[data_col[i]]['x'], lr_result[data_col[i]]['y'], marker = "o")
        axes[1][i].scatter(lr_result[data_col[i]]['x'], lr_result[data_col[i]]['Predictions'], marker = "+")
    axes[2][0].scatter(mlr_result['y'], mlr_result['Predictions'], marker = "o")
    axes[2][0].axline((0, 0), slope=1, color = 'r')
    plt.show()
    

def main():
    data = wrangle("Student_Performance.csv")
    fig, axes = plt.subplots(
        ncols = 5,
        nrows = 3,
        figsize = (12,8)
    )
    linear_regression_result = linear_regression(data)
    linear_regression_summary(linear_regression_result)
    
    multiple_linear_regression_result = multiple_linear_regression(data)
    
    plot(data, linear_regression_result, multiple_linear_regression_result, axes)
    
if __name__ == "__main__":
    main()