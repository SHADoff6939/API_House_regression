from sklearn.ensemble import GradientBoostingRegressor


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GradientBoostingRegressor().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)