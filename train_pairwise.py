# Author: Harsh Kohli
# Date Created: 03-01-2022

import yaml
import tensorflow as tf
from utils.data_utils import load_pairwise_data, create_batches, load_table_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

train_data = load_pairwise_data(config['pretrain_data'])
dev_data = load_pairwise_data(config['pretrain_dev_data'])
tables = load_table_data(config['filtered_tables'])

train_questions, train_contexts, train_titles, train_tables, train_labels = create_batches(train_data, tables,
                                                                                           config['batch_size'])
dev_questions, dev_contexts, dev_titles, dev_tables, dev_labels = create_batches(dev_data, tables, config['batch_size'])
