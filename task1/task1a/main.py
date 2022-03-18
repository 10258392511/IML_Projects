import helpers.config as config


from sklearn.linear_model import Ridge
from helpers.utils import read_data, save_data, rmse_scoring
from helpers.cross_validation import CrossValidator


if __name__ == '__main__':
    # Read Data
    train_path = "./data/train.csv"
    X, y = read_data(train_path)

    # Cross Validation
    classifier = Ridge(tol=1e-10, fit_intercept=False)
    param_grid = {"alpha": config.config_lambda}
    scoring = {"rmse_score": rmse_scoring()}

    cv = CrossValidator(X, y, classifier, param_grid, config.config_K, scoring, refit="rmse_score")
    df_cv = cv.fit()
    rmse_vals = cv.get_rmse()

    # Save Results
    out_path = "./outputs/submission.csv"
    save_data(out_path, rmse_vals.values)
