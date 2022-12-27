import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Data details: listings has totally 75 columns.

# Columns need to be predicted: 61-67(index) which are
# review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,
# review_scores_communication,review_scores_location,review_scores_value

indexes = [62, 63, 64, 65, 66, 67, 61]
predict_index = 61


def read_data() -> DataFrame:
    df = pd.read_csv("listings.csv")
    res = df.iloc[:, indexes]
    return res


def pre_process():
    values_origin = read_data()
    values_without_na = values_origin.dropna()
    x = values_without_na.iloc[:, :6]
    y = values_without_na.iloc[:, -1]

    # Data check:
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    for i in range(6):
        temp = x.iloc[:, i]
        for each in temp:
            if each > max_x:
                max_x = each
            if each < min_x:
                min_x = each
    for each in y:
        if each > max_y:
            max_y = each
        if each < min_y:
            min_y = each
    print("max_x:", max_x)
    print("min_x:", min_x)
    print("max_y:", max_y)
    print("min_y:", min_y)
    # -----------

    poly = PolynomialFeatures(degree=2)
    poly.fit(x)
    x_poly = poly.transform(x)

    return x, y


def linear_model(x, y) -> LinearRegression:
    model = LinearRegression()
    model.fit(x, y)
    return model


def main():
    x, y = pre_process()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = linear_model(x_train, y_train)
    print('Coef:', model.coef_)
    print('Intercept:', model.intercept_)
    print('Accuracy:', model.score(x_test, y_test))


if __name__ == '__main__':
    main()
