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

from annoy import AnnoyIndex
import collections




def get_embeddings(model, test_model_input, user_idx_2_rawid, item_idx_2_rawid, save_path='embedding/'):
    raw_user_id_emb_dict = {}
    raw_item_id_emb_dict = {}
    unique_s = test_model_input['iid'].drop_duplicates()
    test_user_model_input = test_model_input
    all_item_model_input = {"iid": unique_s.values}
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    for i in range(user_embs.shape[0]):
        uid = test_user_model_input['uid'][i]
        raw_user_id_emb_dict[user_idx_2_rawid.get(uid)] = user_embs[i]

    for i in range(item_embs.shape[0]):
        iid = all_item_model_input['iid'][i]
        raw_item_id_emb_dict[item_idx_2_rawid.get(iid)] = item_embs[i]

    pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_youtube_emb.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(save_path + 'doc_youtube_emb.pkl', 'wb'))

    # 读取
    # user_embs_dict = pickle.load(open('embedding/user_youtube_emb.pkl', 'rb'))
    # doc_embs_dict = pickle.load(open('embedding/doc_youtube_emb.pkl', 'rb'))
    return user_embs, item_embs

def get_youtube_recall_res(user_embs, doc_embs, user_idx_2_rawid, doc_idx_2_rawid, topk=10):
    """近邻检索，这里用annoy tree"""
    # 把doc_embs构建成索引树
    f = user_embs.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i, v in enumerate(doc_embs):
        t.add_item(i, v)
    t.build(10)
    # 可以保存该索引树 t.save('annoy.ann')

    # 每个用户向量， 返回最近的TopK个item
    user_recall_items_dict = collections.defaultdict(dict)

    for i, u in enumerate(user_embs):
        print(i,": ", u)
        recall_doc_scores = t.get_nns_by_vector(u, topk, include_distances=True)
        # recall_doc_scores是(([doc_idx], [scores]))， 这里需要转成原始doc的id
        raw_doc_scores = list(recall_doc_scores)
        raw_doc_scores[0] = [doc_idx_2_rawid[i] for i in raw_doc_scores[0]]
        # 转换成实际用户id
        user_recall_items_dict[user_idx_2_rawid[i+1]] = dict(zip(*raw_doc_scores))

    # 默认是分数从小到大排的序， 这里要从大到小
    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}

    # 保存一份
    # pickle.dump(user_recall_doc_dict, open('youtube_u2i_dict.pkl', 'wb'))

    return user_recall_items_dict


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

    #print(user.columns)

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
    #data = ratings
    #print(" data:", data.head(10))

    user_id_enc = data['uid']
    doc_id_enc = data['iid']

    print("*" * 100, " features process")
    feat_embedding_dim = FEAT_CONF.get("emb_dim", 16)
    features_label_file = FEAT_CONF.get("features_label_file","../models/feature_label.pkl")
    feature_process = FeaturesProcess(feature_yaml_path,features_label_file,embedding_dim = 16)
    #print(type(feature_process.features_label['uid']['r_lables']))
    train_model_input, user_feature_columns, item_feature_columns = feature_process.feat_process(data)

    user_idx_2_rawid = feature_process.features_label['uid']['r_lables']
    item_idx_2_rawid = feature_process.features_label['iid']['r_lables']
    train_label = data["rating"]
    print("*" * 100, " training process")
    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    train_counter = Counter(train_model_input['iid'])

    vs = int(item_feature_columns[0].vocabulary_size)
    item_count = [train_counter.get(int(i), 0) for i in range(vs)]
    #sampler_config = NegativeSampler('inbatch', num_sampled=255, item_name="iid", item_count=item_count)
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='iid', item_count=item_count)
    weights_save_file = MODEL_CONF.get('weights_save','../models/ytb_weights.h5')

    # 3.Define Model and train

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)
    try:

        model_embedding_dim = MODEL_CONF.get('embedding_dim', 16)
        print("model_embedding_dim:",model_embedding_dim)

        model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, model_embedding_dim),
                           sampler_config=sampler_config)

        model.compile(optimizer=Adam(learning_rate=0.001), loss=sampledsoftmaxloss)

        model.load_weights(weights_save_file)

    except:
        traceback.print_exc()
    model.save_weights(weights_save_file)

    print("model save over～")

    #print("item_feature_columns:",item_feature_columns)

    """"""
    # 4. Generate user features for testing and full item features for retrieval
    # test_user_model_input = train_model_input
    # user_embs, item_embs = get_embeddings(model, test_user_model_input, user_idx_2_rawid, item_idx_2_rawid, save_path='../models/')
    # print(user_embs.shape)
    # print(item_embs.shape)


    # 5.
    user_embs_dict = pickle.load(open('../models/user_youtube_emb.pkl', 'rb'))
    item_embs_dict = pickle.load(open('../models/doc_youtube_emb.pkl', 'rb'))





