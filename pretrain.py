# Author: Harsh Kohli
# Date Created: 24-11-2021

# from transformers import TapasTokenizer, TFTapasModel
# import pandas as pd
#
# tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
# model = TFTapasModel.from_pretrained('google/tapas-base')
#
# data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
#          'Age': ["56", "45", "59"],
#          'Number of movies': ["87", "53", "69"]
# }
# table = pd.DataFrame.from_dict(data)
# queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]
#
# inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
# #inputs = tokenizer(table=table, padding="max_length", return_tensors="tf")
#
# data2 = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
#          'Age': ["56", "45", "59"],
#          'Number of movies': ["87", "53", "69"]
# }
# table2 = pd.DataFrame.from_dict(data2)
# queries2 = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]
#
# inputs2 = tokenizer(table=table2, queries=queries2, padding="max_length", return_tensors="tf")
#
# model([inputs, inputs2])
#
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state

# import yaml
# import tensorflow as tf
# from transformers import TapasTokenizer
# import pandas as pd
# from utils.data_utils import load_pairwise_data, create_batches, load_table_data
#
# config = yaml.safe_load(open('config.yml', 'r'))
#
# train_data = load_pairwise_data(config['pretrain_data'])
# dev_data = load_pairwise_data(config['pretrain_dev_data'])
# tables = load_table_data(config['filtered_tables'])
#
# model_name = 'google/tapas-base'
# tokenizer = TapasTokenizer.from_pretrained(model_name)
#
# sample = train_data[0]
# table_info = tables[sample['context_id']]
# title, table_data = table_info['pgTitle'], table_info['table_data']
# table_data = pd.DataFrame.from_dict(table_data)
# encoding = tokenizer(table=table_data, queries=title, padding="max_length", return_tensors="tf")
# encoding = {key: tf.squeeze(val, 0) for key, val in encoding.items()}
# print('here')
#
#
#
# class TableDataset:
#
#     def __init__(self, data, tables, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.tables = tables
#
#     def __iter__(self):
#         for idx in range(self.__len__()):
#             sample = self.data[idx]
#             table_info = self.tables[sample['context_id']]
#             title, table_data = table_info['pgTitle'], table_info['table_data']
#             table_data = pd.DataFrame.from_dict(table_data)
#             encoding = self.tokenizer(table=table_data, queries=title, padding="max_length", return_tensors="tf")
#             encoding = {key: tf.squeeze(val, 0) for key, val in encoding.items()}
#             yield encoding['input_ids'], encoding['attention_mask'], encoding['numeric_values'], \
#                   encoding['numeric_values_scale'], encoding['token_type_ids'], encoding['labels']
#             # item = self.data.iloc[idx]
#             # table = pd.read_csv(table_csv_path + item.table_file).astype(str)
#             # encoding = self.tokenizer(table=table,
#             #                       queries=item.question,
#             #                       answer_coordinates=item.answer_coordinates,
#             #                       answer_text=item.answer_text,
#             #                       truncation=True,
#             #                       padding="max_length",
#             #                       return_tensors="tf"
#             # )
#             # # remove the batch dimension which the tokenizer adds by default
#             # encoding = {key: tf.squeeze(val,0) for key, val in encoding.items()}
#             # # add the float_answer which is also required (weak supervision for aggregation case)
#             # encoding["float_answer"] = tf.convert_to_tensor(item.float_answer,dtype=tf.float32)
#             # yield encoding['input_ids'], encoding['attention_mask'], encoding['numeric_values'], \
#             #       encoding['numeric_values_scale'], encoding['token_type_ids'], encoding['labels'], \
#             #       encoding['float_answer']
#
#     def __len__(self):
#        return len(self.data)
#
#     __call__ = __iter__
#
# # data = pd.read_csv(tsv_path, sep='\t')
# train_dataset = TableDataset(train_data, tables, tokenizer)
# output_signature = (
# tf.TensorSpec(shape=(512,), dtype=tf.int32),
# tf.TensorSpec(shape=(512,), dtype=tf.int32),
# tf.TensorSpec(shape=(512,), dtype=tf.float32),
# tf.TensorSpec(shape=(512,), dtype=tf.float32),
# tf.TensorSpec(shape=(512,7), dtype=tf.int32),
# tf.TensorSpec(shape=(512,), dtype=tf.int32))
# train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).batch(32)
#
# import tensorflow as tf
# from transformers import TapasConfig, TFTapasForQuestionAnswering
#
# # this is the default WTQ configuration
# config = TapasConfig(
#            num_aggregation_labels = 4,
#            use_answer_as_supervision = True,
#            answer_loss_cutoff = 0.664694,
#            cell_selection_preference = 0.207951,
#            huber_loss_delta = 0.121194,
#            init_cell_selection_weights_to_zero = True,
#            select_one_column = True,
#            allow_empty_column_selection = False,
#            temperature = 0.0352513,
# )
# model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
#
# for epoch in range(2):  # loop over the dataset multiple times
#    for batch in train_dataloader:
#         # get the inputs;
#         input_ids = batch[0]
#         attention_mask = batch[1]
#         token_type_ids = batch[4]
#         labels = batch[-1]
#         numeric_values = batch[2]
#         numeric_values_scale = batch[3]
#         float_answer = batch[6]
#
#         # forward + backward + optimize
#         with tf.GradientTape() as tape:
#              outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                        labels=labels, numeric_values=numeric_values, numeric_values_scale=numeric_values_scale,
#                        float_answer=float_answer )
#         grads = tape.gradient(outputs.loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained('bert-base-cased')

inputs = tokenizer("Hello, my dog is cute", padding="max_length", truncation=True, return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state
