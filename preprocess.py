# Author: Harsh Kohli
# Date Created: 10-11-2021

import yaml
from utils.data_utils import parse_zipped, filter_useful_tables, write_datafile

config = yaml.safe_load(open('config.yml', 'r'))

train_data, train_tables = parse_zipped(config['train_data_dir'])
dev_data, dev_tables = parse_zipped(config['dev_data_dir'])

all_tables = set()
for table in train_tables:
    all_tables.add(table)
for table in dev_tables:
    all_tables.add(table)

filtered_ids = filter_useful_tables(config['wikitables'], all_tables, config['filtered_tables'])

print('Writing preprocessed train data')
write_datafile(train_data, config['train_data'], filtered_ids)
print('Writing preprocessed dev data')
write_datafile(dev_data, config['dev_data'], filtered_ids)
print('Done preprocessing')
