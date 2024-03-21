import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.ops.init_ops import GlorotUniform, Zeros


class Attention_Eges(Layer):
    def __init__(self, item_nums, l2_reg, seed, **kwargs):
        super(Attention_Eges, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.seed = seed
        self.l2_reg = l2_reg

    def build(self, input_shape):
        super(Attention_Eges, self).build(input_shape)
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("Attention_Eges must have two inputs")

        shape_set = input_shape

        self.feat_nums = shape_set[1][1]
        self.alpha_attention = self.add_weight(
                name='alpha_attention',
                shape=(self.item_nums, self.feat_nums),
                initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=self.seed),
                regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        item_input = inputs[0]
        # (batch_size, feat_nums, embed_size)
        stack_embeds = inputs[1]
        # (batch_size, 1, feat_nums)
        alpha_embeds = tf.nn.embedding_lookup(self.alpha_attention, item_input)
        alpha_embeds = tf.math.exp(alpha_embeds)
        alpha_sum = tf.reduce_sum(alpha_embeds, axis=-1)
        # (batch_size, 1, embed_size)
        merge_embeds = tf.matmul(alpha_embeds, stack_embeds)
        # (batch_size, embed_size), 归一化
        merge_embeds = tf.squeeze(merge_embeds, axis=1)  / alpha_sum
        return merge_embeds

    def compute_mask(self, inputs, mask):
        return None

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1][2])

    def get_config(self):
        config = {'item_nums': self.item_nums, "l2_reg": self.l2_reg, 'seed': self.seed}
        base_config = super(Attention_Eges, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SampledSoftmax(Layer):
    def __init__(self, item_nums, num_sampled, l2_reg, seed, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.num_sampled = num_sampled
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        super(SampledSoftmax, self).build(input_shape)
        embed_size = input_shape[0][1]
        self.softmax_w = self.add_weight(
                                         name="softmax_w",
                                         shape=(self.item_nums, embed_size),
                                         initializer=GlorotUniform(self.seed),
                                         regularizer=l2(self.l2_reg)
                                        )
        self.softmax_b = self.add_weight(
                                         name="softmax_b",
                                         shape=(self.item_nums,),
                                         initializer=Zeros()
                                        )

    def call(self, inputs, training=None, **kwargs):
        input_embed, labels = inputs
        if tf.keras.backend.learning_phase():
            softmax_loss = tf.nn.sampled_softmax_loss(weights=self.softmax_w,
                                                    biases=self.softmax_b,
                                                    labels=labels,
                                                    inputs=input_embed,
                                                    num_sampled=self.num_sampled,
                                                    num_classes=self.item_nums,
                                                    seed=self.seed,
                                                    name="softmax_loss")
        else:
            logits = tf.matmul(input_embed, tf.transpose(self.softmax_w))
            logits = tf.nn.bias_add(logits, self.softmax_b)
            labels_one_hot = tf.one_hot(labels, self.item_nums)
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
                                                                    labels=labels_one_hot,
                                                                    logits=logits)
        return softmax_loss

    def compute_output_shape(self, input_shape):
        return (None,)

    def get_config(self, ):
        config = {'item_nums': self.item_nums, 'num_sampled': self.num_sampled,
                  "l2_reg": self.l2_reg, "seed": self.seed}
        base_config = super(SampledSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(tf.abs(y_true - y_pred))
