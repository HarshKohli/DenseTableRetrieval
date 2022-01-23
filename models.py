# Author: Harsh Kohli
# Date Created: 03-01-2022

import tensorflow as tf
from transformers import TFBertModel, TFTapasModel
from tensorflow.keras import Model


class TableEncoder(Model):
    def __init__(self, config):
        super(TableEncoder, self).__init__()
        self.table_encoder = TFTapasModel.from_pretrained(config['base_model'])
        self.question_encoder = TFBertModel.from_pretrained(config['base_question_encoder'])
        self.margin = config['contrastive_loss_margin']

    def call(self, features, **kwargs):
        input_ids, attention_mask, token_type_ids, question, labels, question_inputs, question_mask, question_type = features
        table_encoding = self.get_table_embedding(input_ids, attention_mask, token_type_ids)
        question_encoding = self.get_question_embedding(question_inputs, question_mask, question_type)
        d = tf.reduce_sum(tf.square(question_encoding - table_encoding), 1)
        d_sqrt = tf.sqrt(d)
        loss = labels * d + (tf.ones(shape=[tf.shape(labels)[0]]) - labels) * tf.square(
            tf.maximum(0., self.margin - d_sqrt))
        loss = 0.5 * tf.reduce_mean(loss)
        return loss

    @tf.function
    def get_table_embedding(self, input_ids, attention_mask, token_type_ids):
        table_encoding = self.table_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        return tf.math.l2_normalize(table_encoding['pooler_output'], axis=1)

    @tf.function
    def get_question_embedding(self, input_ids, attention_mask, token_type_ids):
        question_encoding = self.question_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids)
        return tf.math.l2_normalize(question_encoding['pooler_output'], axis=1)
