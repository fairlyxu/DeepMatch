import numpy as np
from keras_preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat,DenseFeat
from sklearn.preprocessing import MinMaxScaler
import pickle
import traceback


def feature_label_encode():
    # init
    pass
def var_feat_process(data,user_var_feature,item_var_feature,var_features_label_file,embedding_dim=16):
    """
    Args:
        data:
        user_var_feature:
        item_var_feature:
        var_features_label_file:
    Returns:

    """
    var_features_label = {}
    try:
        with open(var_features_label_file, 'rb') as f:
            var_features_label = pickle.load(f)
    except:
        traceback.print_exc()

    # sparse_features_label_file = ""
    # sparse_features_label = {}
    # try:
    #     with open(sparse_features_label_file, 'rb') as f:
    #         sparse_features_label = pickle.load(f)
    #
    # except:
    #     traceback.print_exc()


    def get_var_feature(data, f_name):
        key2index = {}
        if f_name in var_features_label:
            key2index = var_features_label[f_name]

        def split(x):
            if isinstance(x, float):
                return []
            key_ans = x
            for key in key_ans:
                if key not in key2index:
                    # Notice : input value 0 is a special "padding",\
                    # so we do not use 0 to encode valid feature for sequence input
                    key2index[key] = len(key2index) + 1
            return list(map(lambda x: key2index[x], key_ans))

        var_feature = list(map(split, data[f_name].values))
        var_feature_length = np.array(list(map(len, var_feature)))
        max_len = 100 #max(var_feature_length)
        var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
        print(f_name,":var_feature:",var_feature.shape)
        var_features_label[f_name] = key2index
        return (key2index, var_feature, max_len)
    user_var_feature_dict = {}
    user_varlen_feature_columns = []
    for f_name in user_var_feature:
        key2index, var_feature_list, max_len = get_var_feature(data,f_name)
        user_var_feature_dict[f_name] = {"key2index": key2index, "var_feature_list": var_feature_list, "max_len": 100}
        user_varlen_feature_columns.append(
            VarLenSparseFeat(SparseFeat(f_name,vocabulary_size=10000,embedding_dim=embedding_dim), maxlen=100, combiner='mean', length_name=None))

    item_var_feature_dict = {}
    item_varlen_feature_columns = []
    for f_name in item_var_feature:
        key2index, var_feature_list, max_len = get_var_feature(data, f_name)
        item_var_feature_dict[f_name] = {"key2index": key2index, "var_feature_list": var_feature_list,"max_len": max_len}
        item_varlen_feature_columns.append(
            VarLenSparseFeat(SparseFeat(f_name, vocabulary_size=10000,embedding_dim=embedding_dim), maxlen=max_len, combiner='mean', length_name=None))
    with open(var_features_label_file, 'wb') as f:
        pickle.dump(var_features_label, f)

    import json,time
    with open('var_data%d.json' %(time.time()), 'w') as f:
        json.dump(var_features_label, f)
    return user_var_feature_dict,user_varlen_feature_columns,item_var_feature_dict,item_varlen_feature_columns

def sparse_feat_process(data,sparse_features,sparse_features_label_file):
    sparse_features_label = {}
    try:
        with open(sparse_features_label_file, 'rb') as f:
            sparse_features_label = pickle.load(f)

    except:
        traceback.print_exc()

    def sparse_lable_encode(x, feat_name=""):
        if feat_name not in sparse_features_label:
            tmp = {}
            tmp["e_lable"] = {}
            tmp["r_lables"] = {}
            tmp["max"] = 0
            sparse_features_label[feat_name] = tmp
        if x not in sparse_features_label[feat_name]["e_lable"]:
            index = sparse_features_label[feat_name]["max"] + 1
            sparse_features_label[feat_name]["max"] = index
            sparse_features_label[feat_name]["e_lable"][x] = index
            sparse_features_label[feat_name]["r_lables"][index] = x
        return sparse_features_label[feat_name]["e_lable"].get(x)

    for feat in sparse_features:
        data[feat] = data[feat].apply(sparse_lable_encode, feat_name=feat)

    with open(sparse_features_label_file, 'wb') as f:
        pickle.dump(sparse_features_label, f)

    import json, time
    with open('sparse_data%d.json' % (time.time()), 'w') as f:
        json.dump(sparse_features_label, f)

    return data, sparse_features_label


def dense_feat_process(data,dense_features):
    # 稠密特征处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    return data

def feat_process(data,user_var_features,item_var_features,var_features_label_file,
                  user_sparse_features,item_sparse_features,sparse_features_label_file,
                  user_dense_features,item_dense_features,embedding_dim = 16):
    """

    Args:
        data:
        user_var_feature:
        item_var_feature:
        var_features_label_file:
        sparse_features:  user_sparse_features + item_sparse_features
        sparse_features_label_file:
        dense_features:

    Returns:

    """

    data = dense_feat_process(data, user_dense_features + item_dense_features)
    data, sparse_features_label = sparse_feat_process(data,user_sparse_features + item_sparse_features, sparse_features_label_file)

    user_var_feature_dict, user_varlen_feature_columns, item_var_feature_dict, item_varlen_feature_columns = var_feat_process(data, user_var_features, item_var_features, var_features_label_file,embedding_dim)


    # user_feature_columns = [SparseFeat(feat, sparse_features_label[feat]["max"] + 1, use_hash=True, embedding_dim=embedding_dim)
    #                         for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in user_dense_features]
    # item_feature_columns = [SparseFeat(feat, sparse_features_label[feat]["max"] + 1, use_hash=True,embedding_dim=embedding_dim)
    #                         for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in item_dense_features]
    #


    ################################################  hash string
    # user_feature_columns = [SparseFeat(feat, vocabulary_size=1e6, embedding_dim=16, use_hash=True, dtype='string')
    #                         for feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in user_dense_features]
    #
    # item_feature_columns = [SparseFeat(feat, vocabulary_size=1e6, embedding_dim=16, use_hash=True, dtype='string')
    #                         for feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in item_dense_features]


    user_feature_columns = [SparseFeat(feat, vocabulary_size=100000,embedding_dim=4, use_hash=True )
                            for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in user_dense_features]
    item_feature_columns = [SparseFeat(feat, vocabulary_size=100000,embedding_dim=4, use_hash=True )
                            for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in item_dense_features]


    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns

    sparse_features = user_sparse_features + item_sparse_features
    dense_features = user_dense_features + item_dense_features

    # add user history as user_varlen_feature_columns
    # print(sparse_features + dense_features)
    train_model_input = {name: data[name] for name in (sparse_features + dense_features)}

    for f_name in user_var_features:
        train_model_input[f_name] = user_var_feature_dict[f_name]["var_feature_list"]
    for f_name in item_var_features:
        train_model_input[f_name] = item_var_feature_dict[f_name]["var_feature_list"]

    return train_model_input, user_feature_columns,item_feature_columns

