import helpers.config as config

from sklearn.linear_model import Lasso
from helpers.utils import read_data, get_features, rmse_scoring, save_data
from helpers.cross_validation import CrossValidator


if __name__ == '__main__':
    # read data
    train_path = "./data/train.csv"
    X, y = read_data(train_path)
    X_features = get_features(X)

    # cross validation
    # set max_iter to 6000 since convergence is slower than Ridge
    classifier = Lasso(fit_intercept=False, tol=1e-3, max_iter=6000)
    param_grid = {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
    scorings = {"rmse": rmse_scoring()}
    refit = "rmse"
    cv = CrossValidator(X_features, y, classifier, param_grid, config.config_K, scorings, refit=refit)
    df_train = cv.fit()

    # using all training data
    best_alpha = cv.clf.best_params_["alpha"]
    model_all_tr = Lasso(alpha=best_alpha, fit_intercept=False, tol=1e-3)
    model_all_tr.fit(X_features, y)

    # save results
    save_path = "./outputs/submission.csv"
    save_data(save_path, model_all_tr.coef_)
