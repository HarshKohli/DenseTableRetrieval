# Author: Harsh Kohli
# Date Created: 10-11-2021

import yaml
from utils.data_utils import parse_zipped, filter_useful_tables, write_datafile, negative_sampling

config = yaml.safe_load(open('config.yml', 'r'))

train_data, train_tables = parse_zipped(config['train_data_dir'])
dev_data, dev_tables = parse_zipped(config['dev_data_dir'])

all_tables = set()
for table in train_tables:
    all_tables.add(table)
for table in dev_tables:
    all_tables.add(table)

filtered_ids, filtered_tables = filter_useful_tables(config['wikitables'], all_tables, config['filtered_tables'])

print('Starting negative sampling')
train_data, dev_data = negative_sampling(train_data, dev_data, filtered_ids, filtered_tables,
                                         config['random_neg_ratio'], config['hard_neg_ratio'], config['sample_hard'])

print('Writing preprocessed train data')
write_datafile(train_data, config['train_data'], filtered_ids)
print('Writing preprocessed dev data')
write_datafile(dev_data, config['dev_data'], filtered_ids)
print('Done preprocessing')
