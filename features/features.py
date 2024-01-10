import numpy as np
import os
import sys
sys.path.append('../')
from utils import config
from keras_preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat,DenseFeat
from sklearn.preprocessing import MinMaxScaler
import pickle
import traceback


def feature_label_encode(features_label_file,f_key,f_value):
    # init config file
    features_label = {}
    try:
        with open(features_label_file, 'rb') as f:
            features_label = pickle.load(f)
    except:
        traceback.print_exc()
    if f_key not in features_label:
        tmp = {}
        tmp["e_lable"] = {}
        tmp["r_lables"] = {}
        tmp["max"] = 0
        features_label[f_key] = tmp
    if f_value not in features_label[f_key]["e_lable"]:
        index = features_label[f_key]["max"] + 1
        features_label[f_key]["max"] = index
        features_label[f_key]["e_lable"][f_value] = index
        features_label[f_key]["r_lables"][index] = f_value
    # save
    with open(features_label_file, 'wb') as f:
        pickle.dump(features_label, f)
    return features_label[f_key]["e_lable"].get(f_value)


def var_feat_process(data,var_feature={}, embedding_dim=16,features_label_file=""):

    """
    Args:
        data:
        user_var_feature:
        item_var_feature:
        var_features_label_file:
    Returns:
    """

    def get_var_feature(data_list, emb_name,max_len=100):
        var_feature_list = [[feature_label_encode(features_label_file, emb_name, i) for i in sub_list] if isinstance(sub_list, list) else [] for sub_list in data_list]
        var_feature = pad_sequences(var_feature_list, maxlen=max_len, padding='post')
        print(f_name,":var_feature:",var_feature.shape)
        return var_feature

    var_feature_dict = {}
    varlen_feature_columns = []

    for k, v in var_feature.items():
        f_name = k
        vocabulary_size = v.get('size', 1000000)
        max_len = 100
        embedding_dim = embedding_dim
        embedding_name = f_name
        if 'shared' in v and len(v['shared']) > 0:
            embedding_name = v['shared']
        var_feature_list = get_var_feature(data[f_name].values, embedding_name,max_len)
        var_feature_dict[f_name] = {"var_feature_list": var_feature_list}
        varlen_feature_columns.append(VarLenSparseFeat(SparseFeat(f_name, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,embedding_name=embedding_name,use_hash=True), maxlen=max_len, combiner='mean', length_name=None))

    return var_feature_dict, varlen_feature_columns


def sparse_feat_process(data,sparse_features,features_label_file):
    for feat in sparse_features:
        data[feat] = data[feat].map(lambda x: feature_label_encode(features_label_file, feat,x))
    return data


def dense_feat_process(data,dense_features):
    # 稠密特征处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    return data


def feat_process(data, feature_yaml_path="",features_label_file= "",embedding_dim = 16):
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
    user_sparse_f, item_sparse_f, user_var_f, item_var_f, user_dense_f, item_dense_f = config.get_feature_conf(feature_yaml_path)

    # init config file

    user_sparse_features = list(user_sparse_f.keys())
    item_sparse_features = list(item_sparse_f.keys())
    user_dense_features = list(user_dense_f.keys())
    item_dense_features = list(item_dense_f.keys())

    data = dense_feat_process(data, user_dense_features + item_dense_features)
    data = sparse_feat_process(data,user_sparse_features + item_sparse_features, features_label_file)
    user_var_feature_dict, user_varlen_feature_columns = var_feat_process(data, var_feature=user_var_f, embedding_dim=embedding_dim,features_label_file=features_label_file)
    item_var_feature_dict, item_varlen_feature_columns = var_feat_process(data, var_feature=item_var_f, embedding_dim=embedding_dim,features_label_file=features_label_file)


    user_feature_columns = [SparseFeat(name, vocabulary_size=feat.get('size',1000000),embedding_dim=embedding_dim, use_hash=True ) for name, feat in user_sparse_f.items()] \
                           + [DenseFeat(feat, 1, ) for feat in user_dense_features]
    item_feature_columns = [SparseFeat(name, vocabulary_size=feat.get('size',1000000),embedding_dim=embedding_dim, use_hash=True ) for name, feat in item_sparse_f.items()] \
                           + [DenseFeat(feat, 1, ) for feat in item_dense_features]


    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns

    sparse_features = user_sparse_features + item_sparse_features
    dense_features = user_dense_features + item_dense_features

    # add user history as user_varlen_feature_columns
    # print(sparse_features + dense_features)
    train_model_input = {name: data[name] for name in (sparse_features + dense_features)}

    for f_name in user_var_f.keys():
        train_model_input[f_name] = user_var_feature_dict[f_name]["var_feature_list"]
    for f_name in item_var_f.keys():
        train_model_input[f_name] = item_var_feature_dict[f_name]["var_feature_list"]

    return train_model_input, user_feature_columns,item_feature_columns

