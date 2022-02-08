# Author: Harsh Kohli
# Date Created: 10-11-2021

import yaml
from utils.data_utils import parse_zipped, filter_useful_tables, write_datafile, negative_sampling, read_nq_data, \
    parse_nq_tables

config = yaml.safe_load(open('config.yml', 'r'))

finetune_train, finetune_train_tables = read_nq_data(config['finetune_train_data'])
finetune_dev, finetune_dev_tables = read_nq_data(config['finetune_dev_data'])
finetune_test, finetune_test_tables = read_nq_data(config['finetune_test_data'])

train_data, train_tables = parse_zipped(config['pretrain_data_dir'])
dev_data, dev_tables = parse_zipped(config['pretrain_dev_data_dir'])

all_tables = set()
for table in train_tables:
    all_tables.add(table)
for table in dev_tables:
    all_tables.add(table)

nq_ids, nq_tables = parse_nq_tables(config['nq_tables'], config['nq_tables_preprocessed'])
filtered_ids, filtered_tables = filter_useful_tables(config['wikitables'], all_tables, config['filtered_tables'])

print('Total NQ tables are ' + str(len(nq_ids)))

print('Starting negative sampling')
train_data, dev_data, _ = negative_sampling(train_data, dev_data, filtered_ids, filtered_tables,
                                            config['random_neg_ratio'], config['hard_neg_ratio'],
                                            False)
print('Negative Sampling for NQ tables')
nq_train_data, nq_dev_data, nq_test_data = negative_sampling(finetune_train, finetune_dev, nq_ids,
                                                             nq_tables, config['random_neg_ratio'],
                                                             config['hard_neg_ratio'], True,
                                                             test_data=finetune_test)

print('Writing preprocessed train data')
write_datafile(train_data, config['pretrain_data'], filtered_ids)
write_datafile(nq_train_data, config['finetune_train_preprocessed'], nq_ids)
print('Writing preprocessed dev data')
write_datafile(dev_data, config['pretrain_dev_data'], filtered_ids)
write_datafile(nq_dev_data, config['finetune_dev_preprocessed'], nq_ids)
write_datafile(nq_test_data, config['finetune_test_preprocessed'], nq_ids)
print('Done preprocessing')
