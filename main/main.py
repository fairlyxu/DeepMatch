import os
from utils import config,preprocess
from features import features
import tensorflow as tf
from collections import Counter
from tensorflow.python.keras import backend as K
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
import time
import pandas as pd
import pickle
import numpy as np
import traceback
from tensorflow.python.keras.models import save_model,load_model
from deepmatch.layers import custom_objects

if __name__ == "__main__":
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../conf/config.yaml")
    conf = config.get_conf(yaml_path)
    print(conf)
    TRAIN_CONF = conf.get("train")
    FEAT_CONF = conf.get("feat")
    MODEL_CONF = conf.get("model")

    print("*" * 10," sample data process")
    SEQ_LEN = TRAIN_CONF.get('seq_len',50)
    negsample = TRAIN_CONF.get('negsample',10)

    rnames = ['iid', 'uid', 'rating', 'timestamp']
    ratings = pd.read_csv(TRAIN_CONF.get('train_file').get('sample','../data/sample.dat'), sep='\t', header=None, names=rnames)

    # 获取用户特征
    with open(TRAIN_CONF.get('train_file').get('user','../data/user.dat') , 'rb') as file:
        user = pickle.load(file)
    # item 特征
    with open(TRAIN_CONF.get('train_file').get('item','../data/item.dat'), 'rb') as file:
        item = pickle.load(file)

    item["iid"] = item.index
    user["uid"] = user.index

    # datatype 处理
    item = item.infer_objects()
    item['iid'] = item['iid'].astype(np.str)
    item['d_st_cat3'] = item['d_st_cat3'].astype(np.str)
    item['d_st_tag'] = item['d_st_tag']
    item['d_st_from_type'] = item['d_st_from_type'].astype(np.str)
    item['d_st_offer'] = item['d_st_offer'].astype(np.str)

    ratings = ratings.infer_objects()
    ratings['iid'] = ratings['iid'].astype(np.str)
    ratings['uid'] = ratings['uid'].astype(np.str)

    user = user.infer_objects()

    item.index.name = "index"
    user.index.name = "index"

    samples = ratings.iloc[:1000]

    t1 = time.time()
    rating_data_set = preprocess.gen_data_set(samples, SEQ_LEN, negsample)
    print("cost time:", time.time() - t1)
    rating_data = (pd.DataFrame(rating_data_set).iloc[:, 0:3]).rename(columns={0: 'uid', 1: 'iid', 2: 'rating'})
    data = pd.merge(pd.merge(rating_data, user), item)

    print("*" * 10, " features process")
    user_sparse_features = preprocess.split_string(FEAT_CONF.get("user").get("sparse",""))  #['uid', 'u_st_2_country', 'u_st_2_lan', 'u_st_2_brand', 'u_st_2_channel']
    user_dense_features = preprocess.split_string(FEAT_CONF.get("user").get("dense",""))
    user_var_features = preprocess.split_string(FEAT_CONF.get("user").get("var",""))#['u_dy_5_implist', 'u_dy_5_clicklist', 'u_dy_5_installlist']
    item_sparse_features = preprocess.split_string(FEAT_CONF.get("item").get("sparse","")) #['iid', 'd_st_cat3', 'd_st_from_type', 'd_st_offer']

    item_dense_features = preprocess.split_string(FEAT_CONF.get("item").get("dense","")) #['d_dy_4_imppv', 'd_dy_4_clickpv', 'd_dy_4_installpv', 'os_install', 'os_install_ys',
                           #'os_soaring', 'activate_uv', 'yesterday_2_data_count', 'remain_count_2', 'remain_rate_2',
                           #'yesterday_7_data_count', 'remain_count_7', 'remain_rate_7', 'yesterday_30_data_count',
                           #'remain_count_30', 'remain_rate_30', 'remain_rate']
    item_var_features = preprocess.split_string(FEAT_CONF.get("item").get("var","")) #['d_st_tag']
    var_features_label_file = FEAT_CONF.get("var_feat_file","../models/var_features_label.pkl")
    sparse_features_label_file = FEAT_CONF.get("sparse_feat_file","../models/sparse_features_label.pkl")
    feat_embedding_dim = FEAT_CONF.get("emb_dim", 16)

    train_model_input,user_feature_columns,item_feature_columns = features.feat_process(data, user_var_features, item_var_features, var_features_label_file,
                     user_sparse_features, item_sparse_features, sparse_features_label_file,
                     user_dense_features, item_dense_features,feat_embedding_dim )

    train_label = data["rating"]

    print("*" * 10, " training process")
    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    train_counter = Counter(train_model_input['iid'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('inbatch', num_sampled=255, item_name="iid", item_count=item_count)

    weights_save_file = MODEL_CONF.get('weights_save','../models/dssm_weights.h5')

    # 3.Define Model and train

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    try:
        model = load_model(weights_save_file, custom_objects)


    except:
        traceback.print_exc()
        print("init new model")
        model_embedding_dim = MODEL_CONF.get('embedding_dim',32)
        model = DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(128, 64, model_embedding_dim),
                     item_dnn_hidden_units=(64, model_embedding_dim,), loss_type='softmax', sampler_config=sampler_config)

        model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    # training
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=2, verbose=1, validation_split=0.2)
    save_model(model, weights_save_file)

    print(model)


