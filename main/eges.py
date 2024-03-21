from deepctr.feature_column import build_input_features
from deepctr.layers.utils import concat_func
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from main.attention_eges import Attention_Eges,SampledSoftmax
import tensorflow as tf

from deepmatch.inputs import input_from_feature_columns


def EGES(item_feature, side_feature_columns, num_sampled=100, l2_reg_embedding=0.00001, init_std=0.0001,
         seed=1024):
    features = build_input_features( [item_feature] + side_feature_columns)
    labels = Input(shape=(1, ), dtype=tf.int64, name="label")
    # inputs_list = list(features.values()) + [labels]
    features["label"] = labels

    group_embedding_list, dense_value_list = input_from_feature_columns(features, [item_feature] + side_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, seq_mask_zero=False, support_dense=False, support_group=False)
    # concat (batch_size, num_feat, embedding_size)
    concat_embeds = concat_func(group_embedding_list, axis=1)

    # attention
    att_embeds = Attention_Eges(item_feature.vocabulary_size, l2_reg_embedding, seed)([features[item_feature.name], concat_embeds])

    # sample_softmax
    loss = SampledSoftmax(item_feature.vocabulary_size, num_sampled, l2_reg_embedding, seed)([att_embeds, features["label"]])

    model = Model(inputs=features, outputs=loss)
    return model