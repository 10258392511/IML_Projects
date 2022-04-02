import pandas as pd

from helpers.preprocess import summarize_dataset, aggregate_by_mean_and_impute

if __name__ == '__main__':
    train_filename = "data/train_features.csv"
    train_label_filename = "data/train_labels.csv"
    df_train = pd.read_csv(train_filename)
    df_train_label = pd.read_csv(train_label_filename)
    summarize_dataset(df_train, df_train_label)

    # print("Imputing data...")
    # save_path = "data/train_features_agg_imputed.csv"
    # aggregate_by_mean_and_impute(df_train, if_save=True, save_path=save_path)
