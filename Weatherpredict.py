import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
%matplotlib inline

from google.colab import files
uploaded=files.upload()

import io
weather = pd.read_csv(io.BytesIO(uploaded['seattle-weather.csv']))

weather.head()


weather.columns

weather.shape

weather.describe()

weather.isnull().any()

weather.loc[weather['weather'].isnull(),'weather'] = 'rain'

weather.loc[weather['weather'] == 'rain', 'weather'] = 1
weather.loc[weather['weather'] == 'snow', 'weather'] = 0

weather_num = weather[list(weather.dtypes[weather.dtypes!='object'].index)]

weather_y = weather_num.pop('temp_max')
weather_X = weather_num


weather.head()

X_train, X_test, y_train, y_test = train_test_split(weather_X, weather_y, test_size = 0.2, random_state = 4)

X_train.head()

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)


np.mean((prediction-y_test)**2)


pd.DataFrame({'actual': y_test,
              'prediction': prediction,
              'diff': (y_test-prediction)})

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X_train)

poly.fit(X_poly, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)

prediction2 = lin2.predict(poly.fit_transform(X_test))
np.mean((prediction2-y_test)**2)

pd.DataFrame({'actual': y_test,
              'prediction': prediction2,
              'diff': (y_test-prediction2)})

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

prediction3 = regressor.predict(X_test)
np.mean((prediction3-y_test)**2)

pd.DataFrame({'actual': y_test,
              'prediction': prediction3,
              'diff': (y_test-prediction3)})

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth = 10, random_state = 0, n_estimators = 100)
regr.fit(X_train, y_train)

prediction4 = regr.predict(X_test)
np.mean((prediction4-y_test)**2)


pd.DataFrame({'actual': y_test,
              'prediction': prediction4,
              'diff': (y_test-prediction4)})

regr2 = RandomForestRegressor(max_depth = 50, random_state = 0, n_estimators = 100)
regr2.fit(X_train, y_train)


prediction5 = regr.predict(X_test)
np.mean((prediction5-y_test)**2)

pd.DataFrame({'actual': y_test,
              'prediction': prediction5,
              'diff': (y_test-prediction5)})
