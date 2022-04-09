import pandas as pd
import pickle

from helpers.preprocess import summarize_dataset, aggregate_by_mean_and_impute, preprocess_for_transformer

if __name__ == '__main__':
    train_filename = "data/train_features.csv"
    train_label_filename = "data/train_labels.csv"
    test_filename = "data/test_features.csv"

    # data summary
    df_train = pd.read_csv(train_filename)
    df_train_label = pd.read_csv(train_label_filename)
    summarize_dataset(df_train, df_train_label)

    # data imputation
    print("Imputing data...")
    save_path = "data/train_features_agg_imputed_mean.csv"
    aggregate_by_mean_and_impute(df_train, if_save=True, save_path=save_path)

    # # preprocess data for transformer and save it; please refer to dataset_and_train.ipynb
    # save_train_filename = "data/train.pkl"
    # save_test_filename = "data/test.pkl"
    # train_pids, train_data = preprocess_for_transformer(train_filename, train_label_filename)
    # test_pids, test_data = preprocess_for_transformer(test_filename, None)
    #
    # with open(save_train_filename, "wb") as wf:
    #     pickle.dump({"pids": train_pids, "data": train_data}, wf)
    #
    # with open(save_test_filename, "wb") as wf:
    #     pickle.dump({"pids": test_pids, "data": test_data}, wf)
