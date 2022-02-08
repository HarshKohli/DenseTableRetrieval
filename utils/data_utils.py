# Author: Harsh Kohli
# Date Created: 10-11-2021

import os
import json
import gzip
import random
from rank_bm25 import BM25Okapi


def parse_zipped(data_dir):
    all_data, contexts = [], set()
    for filename in os.listdir(data_dir):
        if filename == '.gitignore':
            continue
        data_file = gzip.open(os.path.join(data_dir, filename), 'rb')
        for line in data_file.readlines():
            info = json.loads(line)
            sample = {'question': info['question'], 'context_id': info['context_id'], 'label': '1.0',
                      'answer': info['answer']}
            all_data.append(sample)
            contexts.add(info['context_id'])
        data_file.close()
    return all_data, contexts


def read_nq_data(file_path):
    nq_file = open(file_path, 'r', encoding='utf8')
    all_data, contexts = [], set()
    for line in nq_file.readlines():
        info = json.loads(line)
        for question in info['questions']:
            sample = {'question': question['originalText'], 'context_id': info['table']['tableId'], 'label': '1.0',
                      'answer': question['answer']['answerTexts'][0]}
            all_data.append(sample)
            contexts.add(sample['context_id'])
        nq_file.close()
    return all_data, contexts


def random_negatives(data, table_ids, ratio):
    rand_neg = []
    table_list = list(table_ids)
    for sample in data:
        if sample['label'] == '0.0' or sample['context_id'] not in table_ids:
            continue
        for _ in range(ratio):
            neg_id = random.choice(table_list)
            if neg_id != sample['context_id']:
                new_sample = {'question': sample['question'], 'context_id': neg_id, 'label': '0.0'}
                rand_neg.append(new_sample)

    print('Appending ' + str(len(rand_neg)) + ' random negatives')
    data = data + rand_neg
    random.shuffle(data)
    return data


def get_table_words(tables):
    docs, index_to_id = [], {}
    for index, (_, table) in enumerate(tables.items()):
        doc = table['pgTitle'].lower().split()
        for col_name, rows in table['table_data'].items():
            doc.extend(col_name.lower().split())
            for row in rows:
                doc.extend(row.lower().split())
        docs.append(doc)
        index_to_id[index] = table['id']
    return docs, index_to_id


def is_overlapping(list1, list2):
    for word in list1:
        if word in list2:
            return True
    return False


def hard_negatives(data, tables, ratio):
    texts, index_to_id = get_table_words(tables)
    bm25_obj = BM25Okapi(texts)
    hard_neg = []
    for count, sample in enumerate(data):
        if sample['label'] == '0.0' or sample['context_id'] not in tables:
            continue
        if count % 1 == 0:
            print('Done processing ' + str(count) + ' hard negatives out of ' + str(len(data)))
        question, answer_table, answer_words = sample['question'], sample['context_id'], sample[
            'answer'].lower().split()
        scores = bm25_obj.get_scores(question.lower().split())
        score_tuples = [(score, i) for i, score in enumerate(scores)]
        score_tuples.sort(reverse=True)
        negatives = []
        for i, (_, index) in enumerate(score_tuples):
            table_id = index_to_id[index]
            if table_id != answer_table:
                table_words = texts[index]
                if not is_overlapping(answer_words, table_words):
                    new_sample = {'question': sample['question'], 'context_id': table_id, 'label': '0.0'}
                    negatives.append(new_sample)
                    if len(negatives) == ratio:
                        break
        hard_neg.extend(negatives)

    print('Appending ' + str(len(hard_neg)) + ' hard negatives')
    data = data + hard_neg
    random.shuffle(data)
    return data


def negative_sampling(train_data, dev_data, filtered_ids, filtered_tables, rand_ratio, hard_ratio, use_hard,
                      test_data=None
                      ):
    if use_hard:
        train_data = hard_negatives(train_data, filtered_tables, hard_ratio)
        dev_data = hard_negatives(dev_data, filtered_tables, hard_ratio)
        if test_data:
            test_data = hard_negatives(test_data, filtered_tables, hard_ratio)
    train_data = random_negatives(train_data, filtered_ids, rand_ratio)
    dev_data = random_negatives(dev_data, filtered_ids, rand_ratio)
    if test_data:
        test_data = random_negatives(test_data, filtered_ids, rand_ratio)
    return train_data, dev_data, test_data


def parse_table_info(info):
    sample_table = {}
    sample_table['id'] = info['_id']
    sample_table['pgTitle'] = info['pgTitle']
    sample_table['sectionTitle'] = info['sectionTitle']

    if 'tableCaption' in info:
        sample_table['tableCaption'] = info['tableCaption']
    else:
        sample_table['tableCaption'] = ''

    table_data = {}
    col_info = info['tableHeaders']
    all_cols = []
    for col in col_info[0]:
        all_cols.append(col['text'])
    for header_num in range(1, info['numHeaderRows']):
        headers = col_info[header_num]
        for col_num in range(info['numCols']):
            if all_cols[col_num] != headers[col_num]['text']:
                all_cols[col_num] = all_cols[col_num] + ' ' + headers[col_num]['text']

    for col_name in all_cols:
        table_data[col_name] = []

    for row in info['tableData']:
        for col_name, col in zip(all_cols, row):
            table_data[col_name].append(col['text'])

    sample_table['table_data'] = table_data
    return sample_table


def filter_useful_tables(wikitables, tables, outfile):
    all_tables = open(wikitables, 'r', encoding='utf8')
    filtered = open(outfile, 'w', encoding='utf8')
    matched_count, filtered_ids, filtered_tables = 0, set(), {}
    for num, table in enumerate(all_tables):
        if num % 10000 == 0:
            print('Done processing ' + str(num) + ' tables')
        info = json.loads(table)
        id = info['_id']
        if id in tables:
            parsed_table = parse_table_info(info)
            filtered_tables[parsed_table['id']] = parsed_table
            filtered.write(json.dumps(parsed_table) + '\n')
            matched_count = matched_count + 1
            filtered_ids.add(id)
    all_tables.close()
    filtered.close()
    print('Filtered ' + str(matched_count) + ' tables from Wikipedia tables')
    return filtered_ids, filtered_tables


def parse_nqtables_info(info):
    sample_table = {}
    sample_table['id'] = info['tableId']
    sample_table['pgTitle'] = info['documentTitle']

    all_cols, table_data, num_empty = [], {}, 0
    for col in info['columns']:
        if col['text'] == '':
            table_data['#' + str(num_empty)] = []
            all_cols.append('#' + str(num_empty))
            num_empty = num_empty + 1
        else:
            table_data[col['text']] = []
            all_cols.append(col['text'])

    for row in info['rows']:
        for index, cell in enumerate(row['cells']):
            col_name = all_cols[index]
            table_data[col_name].append(cell['text'])

    sample_table['table_data'] = table_data
    return sample_table


def parse_nq_tables(nq_tables_file, outfile):
    all_tables = open(nq_tables_file, 'r', encoding='utf8')
    all_tables_outfile = open(outfile, 'w', encoding='utf8')
    nq_ids, nq_tables = set(), {}
    for table in all_tables:
        info = json.loads(table)
        parsed_table = parse_nqtables_info(info)
        nq_ids.add(parsed_table['id'])
        nq_tables[parsed_table['id']] = parsed_table
        all_tables_outfile.write(json.dumps(parsed_table) + '\n')
    all_tables.close()
    all_tables_outfile.close()
    return nq_ids, nq_tables


def write_datafile(data, outfile, id_list):
    preprocessed_file = open(outfile, 'w', encoding='utf8')
    count = 0
    for info in data:
        if info['context_id'] in id_list:
            preprocessed_file.write(json.dumps(info) + '\n')
            count = count + 1
    print('Done writing ' + str(count) + ' files')
    preprocessed_file.close()


def load_pairwise_data(filepath):
    data_file = open(filepath, 'r', encoding='utf8')
    pairwise_data = []
    for sample in data_file.readlines():
        pairwise_data.append(json.loads(sample))
    return pairwise_data


def load_table_data(filepath):
    data_file = open(filepath, 'r', encoding='utf8')
    table_data = {}
    for sample in data_file.readlines():
        info = json.loads(sample)
        table_data[info['id']] = {'pgTitle': info['pgTitle'], 'table_data': info['table_data']}
    return table_data


def write_metrics(filename, metrics_dict, epoch):
    metric_file = open(filename, 'a', encoding='utf8')
    if epoch == 0:
        for index, (metric, value) in enumerate(metrics_dict.items()):
            if index == len(metrics_dict) - 1:
                metric_file.write(metric + '\n')
            else:
                metric_file.write(metric + ',')
    for index, (metric, value) in enumerate(metrics_dict.items()):
        if index == len(metrics_dict) - 1:
            metric_file.write(str(value) + '\n')
        else:
            metric_file.write(str(value) + ',')
        print(metric + ' at the end of epoch ' + str(epoch) + ' is ' + str(value))
    metric_file.close()


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
