import os
import sys
sys.path.append('../')

from util import config, preprocess
from features.features import FeaturesProcess
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
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../conf/config.yaml")
    feature_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../conf/features.yaml")
    conf = config.get_conf(yaml_path)
    TRAIN_CONF = conf.get("train")
    FEAT_CONF = conf.get("feat")
    MODEL_CONF = conf.get("model")

    if TRAIN_CONF["init"]:
        print("new start")

    print("*" * 100," sample data process")
    SEQ_LEN = TRAIN_CONF.get('seq_len',50)
    negsample = TRAIN_CONF.get('negsample',10)

    train_batch_size=TRAIN_CONF.get('batch_size',256)
    train_epochs = 1#TRAIN_CONF.get('epochs', 10)

    #获取样本数据
    rnames = ['uid', 'iid', 'rating']
    ratings = pd.read_csv(TRAIN_CONF.get('train_file').get('sample','../data/sample.dat'), sep='\t', header=None, names=rnames)

    # 获取用户特征
    with open(TRAIN_CONF.get('train_file').get('user','../data/user.dat'), 'rb') as file:
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
    ratings = ratings.iloc[:1000]
    data = pd.merge(pd.merge(ratings, user, how = "left"), item, how = "left")
    data = ratings
    print("data:",data.head(10))

    print("*" * 100, " features process")
    feat_embedding_dim = FEAT_CONF.get("emb_dim", 16)

    features_label_file = FEAT_CONF.get("features_label_file","../models/feature_label.pkl")

    feature_process = FeaturesProcess(feature_yaml_path,features_label_file,embedding_dim = 16)

    train_model_input, user_feature_columns, item_feature_columns = feature_process.feat_process(data)
    print("item_feature_columns:",item_feature_columns)

    train_label = data["rating"]
    print("*" * 100, " training process")
    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    train_counter = Counter(train_model_input['iid'])

    vs = int(item_feature_columns[0].vocabulary_size)
    item_count = [train_counter.get(int(i), 0) for i in range(vs)]
    sampler_config = NegativeSampler('inbatch', num_sampled=255, item_name="iid", item_count=item_count)

    weights_save_file = MODEL_CONF.get('weights_save','../models/dssm_weights.h5')

    # 3.Define Model and train

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    try:
        model_embedding_dim = MODEL_CONF.get('embedding_dim', 32)
        model = DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(128, 64, model_embedding_dim),
                     item_dnn_hidden_units=(64, model_embedding_dim,), loss_type='softmax',
                     sampler_config=sampler_config)


        model.compile(optimizer=Adam(learning_rate=0.001), loss=sampledsoftmaxloss)

        if len(weights_save_file) > 0 and os.path.exists(weights_save_file):
            print(" load weight: ")
            model.load_weights(weights_save_file)
        print(model.summary())
        # training
        history = model.fit(train_model_input, train_label, batch_size=train_batch_size, epochs=train_epochs, verbose=1,
                            validation_split=0.2)

    except:
        traceback.print_exc()
    model.save_weights(weights_save_file)
    print("user_embedding:",model.user_embedding)
    print("item_embedding",model.item_embedding)
    print("model save over～")

    #print("item_feature_columns:",item_feature_columns)

    # 4. Generate user features for testing and full item features for retrieval
    print(item['d_st_2_did'].values)
    #test_user_model_input = item_feature_columns
    all_item_model_input = {"iid": data['iid'].values}

    #user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    #user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    #print(user_embs.shape)
    print(item_embs.shape)
    print(item_embs)




