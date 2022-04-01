import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

dataset_training = pd.read_csv("../datasets/HousePrices/train.csv")

dataset_test = pd.read_csv("../datasets/HousePrices/test.csv")

dev_training_set = train_test_split(dataset_training, test_size= 0.2)

models = [LinearRegression(), SVR(kernel='rbf', degree=2),
          RandomForestRegressor(n_estimators=5000), KNeighborsRegressor(n_neighbors=10),
          GradientBoostingRegressor(learning_rate=0.01, n_estimators=5000)]

def Cleaner(dataset, argument = 'train'):
    columns_to_drop = []
    cleaned_data = dataset.loc[:, dataset.isnull().sum(axis=0) < 500]
    correlation_matrix = dataset_training.corr()
    to_remove = correlation_matrix[(correlation_matrix['SalePrice'].sort_values(ascending=False) < 0.03) & (correlation_matrix['SalePrice'].sort_values(ascending=False) > -0.03)]
    columns_to_drop.extend(to_remove.index.tolist())
    # print(correlation_matrix['SalePrice'].sort_values(ascending=False))
    # print(type(to_remove.index.values.tolist()))
    if argument == 'train':
        labels = cleaned_data['SalePrice']
        columns_to_drop.append('SalePrice')
        data = cleaned_data.drop(columns_to_drop, axis=1)
    else:
        data = cleaned_data.drop(columns_to_drop, axis=1)
        labels = np.array([])
    data.hist()
    return data, labels

def prediction(dataset, labels, cv_data, cv_label, test_data):
    plt.show()
    labels = labels.to_numpy()
    cv_label = cv_label.to_numpy()
    numeric_features = dataset.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = dataset.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder())])

    data_pipeline = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    predictor_pipeline = Pipeline(steps=[
        ('preprocessor', data_pipeline),
        ('regressor', LinearRegression()),
    ])
    count = 0
    score_list = {}
    for regress in models:
        count+=1
        predictor_pipeline.set_params(regressor=regress)
        predictions = predictor_pipeline.fit(dataset, labels).predict(cv_data)
        results = mean_squared_error(predictions, cv_label)
        score_list[regress] = results
    best_regressor = min(score_list, key=score_list.get)
    print(best_regressor)
    predictor_pipeline.set_params(regressor=best_regressor)
    prediction_on_test = predictor_pipeline.fit(dataset, labels).predict(test_data)
    test_labels = pd.DataFrame(prediction_on_test)
    test_labels.columns = ['SalePrice']
    test_labels['SalePrice'] = (test_labels['SalePrice']).astype(int)
    test_data = pd.concat([test_data, test_labels], axis=1)
    return test_data
data, label = Cleaner(dataset_training, 'train')
cv_data, cv_label = Cleaner(dataset_training, 'train')
test_data, test_label = Cleaner(dataset_test, 'test')
results = prediction(data, label, cv_data, cv_label, test_data)
results = pd.concat([dataset_test["Id"], results], axis = 1)

df = pd.DataFrame(results, columns= ['Id', 'SalePrice'])
df.to_csv (r'.\export_House_Prices.csv', index = False, header=True)