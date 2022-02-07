# Author: Harsh Kohli
# Date Created: 03-01-2022

import tensorflow as tf
from transformers import TFBertModel, TFTapasModel
from tensorflow.keras import Model


class TableEncoder(Model):
    def __init__(self, conf):
        super(TableEncoder, self).__init__()
        self.table_encoder = TFTapasModel.from_pretrained(conf['base_model'])
        self.question_encoder = TFBertModel.from_pretrained(conf['base_question_encoder'])
        self.margin = conf['contrastive_loss_margin']

    def call(self, inputs, training=True):
        input_ids, attention_mask, token_type_ids, question, labels, question_inputs, question_mask, question_type = inputs
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

    def get_config(self):
        return {"table_encoder": self.table_encoder, "question_encoder": self.question_encoder, "margin": self.margin}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
