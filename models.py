# Author: Harsh Kohli
# Date Created: 03-01-2022

from transformers import TFBertModel, TFTapasModel
from tensorflow.keras import Model


class TableEncoder(Model):
    def __init__(self, config):
        super(TableEncoder, self).__init__()
        self.table_encoder = TFTapasModel.from_pretrained(config['base_model'])
        self.question_encoder = TFBertModel.from_pretrained(config['base_question_encoder'])

    def call(self, features, **kwargs):
        input_ids, attention_mask, token_type_ids, question, labels, question_inputs, question_mask, question_type = features
        table_encoding = self.table_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        question_encoding = self.question_encoder(input_ids=question_inputs, attention_mask=question_mask,
                                                  token_type_ids=question_type)

        return table_encoding['pooler_output']
