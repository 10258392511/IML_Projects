import pandas as pd
import helpers.configs as configs
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from helpers.preprocess import min_max_features


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from joblib import dump, load


def subtask1():
    # TODO: fill in training procedure; returns pd.DataFrame
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    # read training data
    train_data = pd.read_csv(train_path)
    train_data = train_data.fillna(train_data.mean())

    # extract features
    df_train_agg_max = train_data.groupby("pid").agg("max")
    df_train_agg_max = df_train_agg_max.drop(columns=['Time', 'Age'])
    df_train_agg_max.columns = df_train_agg_max.columns + ['_Max']


    df_train_agg_mean = train_data.groupby("pid").agg("mean")
    df_train_agg_mean = df_train_agg_mean.drop(columns=['Time'])
    df_train_agg_mean.columns = df_train_agg_mean.columns + ['_Mean']


    df_train_agg_min = train_data.groupby("pid").agg("min")
    df_train_agg_min = df_train_agg_min.drop(columns=['Time', 'Age'])
    df_train_agg_min.columns = df_train_agg_min.columns + ['_Min']

    features = df_train_agg_max.join(df_train_agg_mean, on='pid')
    features = features.join(df_train_agg_min, on='pid')

    ## add the variance of each feature
    var_features = []
    pids = train_data.pid.unique()
    for pid in pids[:]:
        t = train_data.loc[train_data['pid'] == pid]
        var = t.var()
        var_features.append(var[3:])

    var_df = pd.DataFrame(var_features)
    pids = train_data.pid.unique()
    var_df['pid'] = pids[:]
    var_df = var_df.set_index('pid')
    var_df.columns = var_df.columns + ['_Variance']
    var_df.sort_index(ascending=True)

    features = features.join(var_df, on='pid')
    # feature ending

    # process label
    train_label = pd.read_csv(train_label_path)
    train_label.set_index('pid', inplace=True)
    label_list = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    labels = train_label[label_list]

    #process test data
    test_data = pd.read_csv(test_path)
    test_data = test_data.fillna(train_data.mean())
    df_test_agg_max = test_data.groupby("pid").agg("max")
    df_test_agg_max = df_test_agg_max.drop(columns=['Time', 'Age'])
    df_test_agg_max.columns = df_test_agg_max.columns + ['_Max']


    df_test_agg_mean = test_data.groupby("pid").agg("mean")
    df_test_agg_mean = df_test_agg_mean.drop(columns=['Time'])
    df_test_agg_mean.columns = df_test_agg_mean.columns + ['_Mean']


    df_test_agg_min = test_data.groupby("pid").agg("min")
    df_test_agg_min = df_test_agg_min.drop(columns=['Time', 'Age'])
    df_test_agg_min.columns = df_test_agg_min.columns + ['_Min']

    features_test = df_test_agg_max.join(df_test_agg_mean, on='pid')
    features_test = features_test.join(df_test_agg_min, on='pid')
    ## add the variance of each feature
    var_features_test = []
    pids = test_data.pid.unique()
    for pid in pids[:]:
        t = test_data.loc[test_data['pid'] == pid]
        var = t.var()
        var_features_test.append(var[3:])

    var_test = pd.DataFrame(var_features_test)
    pids = test_data.pid.unique()
    var_test['pid'] = pids[:]
    var_test = var_test.set_index('pid')
    var_test.columns = var_test.columns + ['_Variance']
    var_test.sort_index(ascending=True)

    features_test = features_test.join(var_test, on='pid')

    result_path = "data/subtask1(1).csv"
    df_result = pd.read_csv(result_path)
    train_label.set_index('pid', inplace=True)

    #prepare model
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(features)
    X = data_rescaled
    Ys = []
    for label in label_list:
        y = labels[label].to_numpy()
        Ys.append(y)

    res = []
    for i in range(len(Ys)):
        print("training mdoel for ", label_list[i])
        y = Ys[i]
        # print(np.count_nonzero(y) / len(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=114)
        # print(np.count_nonzero(y_test) / len(y_test))
        clf = SVC(gamma='auto', probability=True, class_weight='balanced')
        clf.fit(X_train, y_train)
        predict = clf.predict_proba(X_test)

        # using imblearn
        # pipeline = Pipeline(steps=[("upsample", configs.conf_over_sampler_args["class"](**configs.conf_over_sampler_args["args"])),
        #                        ("normalizer", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        #                       ("svc", SVC(probability=True))])
        # pipeline.fit(X_train, y_train)
        # predict = pipeline.predict_proba(X_train)
        fpr, tpr, _ = roc_curve(y_test, predict[:, 1])
        auc_score = auc(fpr, tpr)
        res.append(auc_score)
        print("ROC score for validation set: ", auc_score)
        dump(clf, label_list[i]+'.joblib') 
        # print("predict on test set")
        # test_pred = clf.predict_proba(test_rescaled)

    # predict on test data
    preds = []
    test_rescaled = scaler.fit_transform(features_test)
    for i in range(len(label_list)):
        print("predict for ", label_list[i])
        model_name = label_list[i]+ '.joblib'
        clf = load(model_name) 
        test_pred = clf.predict_proba(test_rescaled)
        preds.append(test_pred[:, 1])

    pred_np = np.array(preds)
    df_result = pd.DataFrame(data=pred_np.T, columns=label_list)
    pids = test_data.pid.sort_values().unique()
    df_result.insert(loc=0,
      column='pid',
      value=pids)

    return df_result


def subtask2():
    label_subtask2 = "LABEL_Sepsis"
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svc", SVC(**configs.conf_SVC_args))
    ])

    # from grid search
    best_params = {
        "svc__C": 0.001,
        "svc__gamma": 0.011,
        "svc__class_weight": {0: 1, 1: 9}
    }

    pipeline.set_params(**best_params)
    print("subtask2 training...")
    pipeline.fit(df_train.values, df_train_labels[label_subtask2].values)

    test_pids = df_test_orig["pid"].sort_values().unique()
    y_pred = pipeline.predict_proba(df_test.values)
    y_pred = y_pred[:, 1]
    data_out = {"pid": test_pids, label_subtask2: y_pred}
    df_out = pd.DataFrame(data=data_out)

    return df_out


def subtask3():
    labels_subtask3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    # from grid search
    best_params = [
        {
            "svr__C": 300,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
         },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        }
    ]

    test_pids = df_test_orig["pid"].sort_values().unique()
    svr_pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svr", SVR())
    ])

    data_out = {"pid": test_pids}
    for label, best_param in zip(labels_subtask3, best_params):
        svr_pipeline.set_params(**best_param)
        print(f"subtask3 training: {label}...")
        svr_pipeline.fit(df_train.values, df_train_labels[label].values)
        y_pred = svr_pipeline.predict(df_test.values)
        data_out[label] = y_pred

    df_out = pd.DataFrame(data=data_out)

    return df_out


if __name__ == '__main__':
    df_subtask1 = subtask1()
    df_subtask2 = subtask2()
    df_subtask3 = subtask3()
    df_out = pd.concat([df_subtask1, df_subtask2.drop("pid", axis=1), df_subtask3.drop("pid", axis=1)], axis=1)
    out_path = "data/task2_results.zip"
    df_compression_args = dict(method="zip", archive_name="task2_results.csv")
    df_out.to_csv(out_path, float_format="%.3f", index=False, compression=df_compression_args)

