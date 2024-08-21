#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
desc: 这是计算相似度的脚本
author: fairy
"""
import pickle
import math
from annoy import AnnoyIndex
import argparse
#import redis

import time


class TopKSearch(object):
    def __init__(self, input_vecs):
        self.tree_num = 40
        self.input_vecs = input_vecs
        self.size = 0
        self.embeddings = None
        self.vect_tree = None
        self.id2idx = None

    def load_doc_vec(self):
        """
        加载向量
        :param model:
        :return:
        """

        ret = {}
        size = len(list(self.input_vecs.values())[0])
        for k, v in self.input_vecs.items():
            tmp = self.__normlize(list(map(lambda _: float(_), v)))
            if tmp is not None:
                ret[k] = tmp

        self.embeddings = ret
        self.size = size

        # return ret, size

    def build_tree(self):
        """
        建立树
        :param size:
        :param vecs:
        :return:
        """
        id2idx = {}
        count = 0
        vt = AnnoyIndex(self.size, metric='euclidean')
        for (_id, vec) in self.embeddings.items():
            vt.add_item(count, vec)
            id2idx[count] = _id
            count += 1
        vt.build(self.tree_num)  # 树的个数
        self.vect_tree = vt
        self.id2idx = id2idx

    def __normlize(self, float_list):
        """
        归一化
        :param float_list:
        :return:
        """
        s = 0.0
        for v in float_list:
            s += v * v
        if s <= 0.0:
            return None
        s = math.sqrt(s)
        ret = []
        for v in float_list:
            ret.append(v / s)
        return ret

    def gen_topk(self, items_ids, outfile,topk=20,):
        """
        找相似item
        :param vecs:
        :param id2idx:
        :param vt:
        :return:
        """
        f = open(outfile, 'w')
        K = topk * 3
        i = 0
        sim_arr = []
        res = {}
        for (_id, vec) in self.embeddings.items():
            if _id not in items_ids:
                continue
            tmp = self.vect_tree.get_nns_by_vector(vec, K, search_k=-1, include_distances=True)
            sim = []
            for (idx, score) in zip(tmp[0], tmp[1]):
                if idx in self.id2idx:
                    s = 1.0 - score / 2.0
                    if s < 0.5:
                        continue
                # 过滤掉本身
                if self.id2idx[idx] == _id:
                    continue
                sim.append('%s' % (self.id2idx[idx]))
            if len(sim) > 0:
                res[_id] = ','.join(sim[:topk])
                f.write('%s\t%s\n' % (_id, ','.join(sim[:topk])))
                sim_arr.append({'uid': _id, 'sim': ','.join(sim[:topk])})
                i += 1

        print("push to redis, item length", len(sim_arr))
        f.close()
        return sim_arr



    def get_youtube_recall_res(self, embeddings, outfile="../output/u2i_sim.csv",topk=20,):
        """
        找相似item
        :param vecs:
        :param id2idx:
        :param vt:
        :return:
        """
        f = open(outfile, 'w')
        K = topk * 3
        i = 0
        sim_arr = []
        res = {}
        for (_id, vec) in embeddings.items():
            tmp = self.vect_tree.get_nns_by_vector(vec, K, search_k=-1, include_distances=True)
            sim = []
            for (idx, score) in zip(tmp[0], tmp[1]):
                if idx in self.id2idx:
                    s = 1.0 - score / 2.0
                    if s < 0.5:
                        continue
                # 过滤掉本身
                if self.id2idx[idx] == _id:
                    continue
                sim.append('%s' % (self.id2idx[idx]))
            if len(sim) > 0:
                res[_id] = ','.join(sim[:topk])
                f.write('%s\t%s\n' % (_id, ','.join(sim[:topk])))
                sim_arr.append({_id : ','.join(sim[:topk])})
                i += 1

        print("push to redis, item length", len(sim_arr))
        f.close()
        return sim_arr

def push_to_redis(datas,key,host,db):
    """
    param
    datas: {'uid': 'simlist'}
    return:
    """

    REDIS_URL = "redis://%s:6379/%s"%(host,db)
    conn = redis.from_url(REDIS_URL)
    try:
        with conn.pipeline(transaction=False) as p:
            p.hmset(key, datas)
            p.execute()
    except Exception as e:
        print(e)




def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="init config")
    parser.add_argument('--model_file', nargs='?', default='../output/app.emb',
                        help='graph_model')
    parser.add_argument('--output', nargs='?', default='../output/sim.csv',
                        help='result output')
    parser.add_argument('--redis_host', nargs='?', default='engine-ire-01.htryhl.0001.euw1.cache.amazonaws.com',
                        help='redis_host')
    parser.add_argument('--redis_db', nargs='?', default='engine-ire-01.htryhl.0001.euw1.cache.amazonaws.com',
                        help=4)
    return parser.parse_args()


def main(args):
    """
    main fun
    :return:
    """
    # 相似度计算模型
    model_file = "../models/doc_youtube_emb.pkl" #  args.model_file
    with open(model_file, 'rb') as file:
        embeddings = pickle.load(file)
    user_model_file = "../models/user_youtube_emb.pkl"  # args.model_file
    with open(user_model_file, 'rb') as file:
        user_embeddings = pickle.load(file)

    if len(embeddings) < 1:
        return
    print(type(embeddings))

    tk = TopKSearch(embeddings)
    tk.load_doc_vec()
    tk.build_tree()
    items_ids = embeddings.keys()
    items_size = len(items_ids)
    print("items length is %d" % (items_size))
    if items_size < 1:
        return
    #datas = tk.gen_topk(items_ids, args.output)
    datas = tk.get_youtube_recall_res(embeddings,outfile="../output/i2i_sim.csv" )
    datas = tk.get_youtube_recall_res(user_embeddings,outfile="../output/u2i_sim.csv" )
    #push_to_redis(datas, args.redis_host, args.redis_db)


if __name__ == '__main__':
    st = time.time()
    args = parse_args()
    print(args)
    main(args)
    print(" all work done ,cost %s" % (time.time() - st))
