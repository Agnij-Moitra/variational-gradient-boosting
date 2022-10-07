
import numpy as np
import pandas as pd
from numba import prange
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, SGDRegressor, LassoLars, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from xgboost import XGBRegressor, XGBRFRegressor
import time
import math


class VGBRegressor(object):
    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        custom_loss=None,
        early_stopping: bool = False,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        complexity: bool = False,
        custom_models: list = None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss = custom_loss
        self.early_stopping = early_stopping
        self._X = None
        self._y = None
        if custom_models:
            self._models = custom_models
        else:
            if complexity:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor, ExtraTreesRegressor,
                                RadiusNeighborsRegressor, ElasticNet, LassoLars, Lasso, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                                BaggingRegressor, SVR, NuSVR, XGBRegressor, XGBRFRegressor, SGDRegressor, KernelRidge, MLPRegressor)
            else:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
                                RadiusNeighborsRegressor, ElasticNet, LassoLars, Lasso, SGDRegressor, BaggingRegressor)
        self._ensemble = []

    def _metrics(self, vt, vp, model, time=None):
        if self.custom_loss:
            return {'model': model, 'time': time, 'loss': self.custom_loss(vt, vp)}
        return {"model": model, "time": time, "loss": mean_absolute_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it: bool = False):
        model = Pipeline([
            ('scaler1', RobustScaler()),
            ('model', model_name())
        ])
        model_name()
        if time_it:
            begin = time.time()
            model.fit(X, y)
            end = time.time()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, X, y, model_name):
        Xt, Xv, yt, yv = train_test_split(X, y)
        results = self._create_model(Xt, yt, model_name, time_it=False)
        model, time = results[0], results[1]
        return self._metrics(yv,
                             model.predict(Xv), model, time)

    def _get_results(self, X, y) -> list:
        results = []
        for i in self._models:
            try:
                results.append(self._get_metrics(X, y, i))
            except:
                pass
        return results

    def fit(self, X_train, y_train):
        # base model: mean
        # computer residuals: y - y hat
        # for n_estimators: a) y = prev residuals && residuals * learning rate
        # ada boost and adaptive scaling for learning rates

        preds = pd.DataFrame(
            data={'p0': np.full((len(y_train)), y_train.mean(skipna=True))})
        residuals = pd.DataFrame(
            data={'r0': y_train - y_train.mean(skipna=True)})

        for i in prange(1, self.n_estimators + 1):
            y = residuals[f'r{i - 1}']
            results = self._get_results(X_train, y)
            min_loss = min(results, key=lambda d: d.get(
                "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
            min_model = [i['model']
                         for i in results if min_loss == i['loss']][0]
            residuals[f'r{i}'] = min_model.predict(
                X_train) * self.learning_rate
            X_train[f'r{i}'] = residuals[f'r{i - 1}']
            self._ensemble.append(min_model)
        return self._ensemble, residuals
