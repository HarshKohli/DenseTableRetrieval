# Author: Harsh Kohli
# Date Created: 10-11-2021

import os
import json
import gzip


def parse_zipped(data_dir):
    all_data, contexts = [], set()
    for filename in os.listdir(data_dir):
        if filename == '.gitignore':
            continue
        data_file = gzip.open(os.path.join(data_dir, filename), 'rb')
        for line in data_file.readlines():
            info = json.loads(line)
            all_data.append(info)
            contexts.add(info['context_id'])
        data_file.close()
    return all_data, contexts


def filter_useful_tables(wikitables, tables, outfile):
    all_tables = open(wikitables, 'r', encoding='utf8')
    filtered = open(outfile, 'w', encoding='utf8')
    matched_count, filtered_ids = 0, set()
    for num, table in enumerate(all_tables):
        if num % 10000 == 0:
            print('Done processing ' + str(num) + ' tables')
        info = json.loads(table)
        id = info['_id']
        if id in tables:
            filtered.write(table)
            matched_count = matched_count + 1
            filtered_ids.add(id)
    all_tables.close()
    filtered.close()
    print('Filtered ' + str(matched_count) + ' tables from Wikipedia tables')
    return filtered_ids

def write_datafile(data, outfile, id_list):
    preprocessed_file = open(outfile, 'w', encoding='utf8')
    count = 0
    for info in data:
        if info['context_id'] in id_list:
            preprocessed_file.write(json.dumps(info))
            count = count + 1
    print('Done writting ' + str(count) + ' files')
    preprocessed_file.close()
