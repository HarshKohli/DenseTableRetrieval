# Author: Harsh Kohli
# Date Created: 04-01-2022

import pandas as pd
import tensorflow as tf


class TableDataset:

    def __init__(self, data, tables, tokenizer, question_tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.question_tokenizer = question_tokenizer
        self.tables = tables

    def __iter__(self):
        for idx in range(self.__len__()):
            try:
                sample = self.data[idx]
                table_info = self.tables[sample['context_id']]
                title, table_data = table_info['pgTitle'], table_info['table_data']
                table_data = pd.DataFrame.from_dict(table_data)
                encoding = self.tokenizer(table=table_data, queries=title, padding="max_length", truncation=True,
                                          return_tensors="tf")
                encoding = {key: tf.squeeze(val, 0) for key, val in encoding.items()}
                question_encoding = self.question_tokenizer(sample['question'], padding="max_length", truncation=True,
                                                            return_tensors="tf")
                question_input = {key: tf.squeeze(val, 0) for key, val in question_encoding.data.items()}
                yield encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'], sample[
                    'question'], float(sample['label']), question_input['input_ids'], question_input['attention_mask'], \
                      question_input['token_type_ids']
            except:
                continue

    def __len__(self):
        return len(self.data)

    __call__ = __iter__
