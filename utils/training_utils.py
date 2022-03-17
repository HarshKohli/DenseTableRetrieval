# Author: Harsh Kohli
# Date Created: 04-01-2022

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from annoy import AnnoyIndex


class PairwiseDataset:

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
                                                            return_tensors="tf", max_length=128)
                question_input = {key: tf.squeeze(val, 0) for key, val in question_encoding.data.items()}
                yield encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'], sample[
                    'question'], float(sample['label']), question_input['input_ids'], question_input['attention_mask'], \
                      question_input['token_type_ids'], sample['context_id']
            except:
                continue

    def __len__(self):
        return len(self.data)

    __call__ = __iter__


class TablesDataset:

    def __init__(self, tables, tokenizer):
        self.tables = tables
        self.tokenizer = tokenizer

    def __iter__(self):
        for table_id, table_info in self.tables.items():
            title, table_data = table_info['pgTitle'], table_info['table_data']
            table_data = pd.DataFrame.from_dict(table_data)
            encoding = self.tokenizer(table=table_data, queries=title, padding="max_length", truncation=True,
                                      return_tensors="tf")
            encoding = {key: tf.squeeze(val, 0) for key, val in encoding.items()}
            yield encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'], table_id

    def __len__(self):
        return len(self.tables)

    __call__ = __iter__


def get_dev_metrics(dev_tables, dev_questions, dev_labels):
    metrics_dict, tp, tn, fp, fn = {}, 0, 0, 0, 0
    for table, question, label in zip(dev_tables, dev_questions, dev_labels):
        sim = cosine_similarity([table], [question])
        if label == 0:
            if sim > 0.5:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if sim < 0.5:
                tn = tn + 1
            else:
                fp = fp + 1
    if (tp + fp) > 0:
        metrics_dict['precision'] = tp / (tp + fp)
        metrics_dict['recall'] = tp / (tp + fn)
        metrics_dict['f_score'] = 2 * metrics_dict['precision'] * metrics_dict['recall'] / (
                metrics_dict['precision'] + metrics_dict['recall'])
    else:
        metrics_dict['precision'] = 0.0
        metrics_dict['recall'] = 0.0
        metrics_dict['f_score'] = 0.0
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics_dict['tp'] = tp
    metrics_dict['tn'] = tn
    metrics_dict['fp'] = fp
    metrics_dict['fn'] = fn
    return metrics_dict


def get_query_table_ranks(sample_info_dict, id_to_index, index_file, dim):
    u = AnnoyIndex(dim, 'angular')
    u.load(index_file)
    ranks = []
    for sentence, info in sample_info_dict.items():
        table_id, embedding = info['table_id'], info['embedding']
        table_index = id_to_index[table_id]
        closest_tables = u.get_nns_by_vector(embedding, 1000000)
        rank = closest_tables.index(table_index) + 1
        ranks.append(rank)
    return ranks


def index_embeddings(id2_embed, index_file, dim):
    t = AnnoyIndex(dim, 'angular')
    for id, embedding in id2_embed.items():
        t.add_item(id, embedding)
    t.build(30)
    dir_name = os.path.dirname(index_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    t.save(index_file)


def get_metrics(ranks):
    dp = [0 for _ in range(max(ranks) + 1)]
    rr = 0
    for rank in ranks:
        dp[rank] = dp[rank] + 1
        rr = rr + 1 / float(rank)
    mrr = rr / len(ranks)
    for index in range(1, len(dp)):
        dp[index] = dp[index] + dp[index - 1]
    total = dp[-1]
    p_scores = []
    for num in dp:
        p_scores.append(num / float(total))
    return mrr, p_scores


def metrics_logger(sample_info_dict, id_to_index, index_file, dim, eval_set, find_p, outfile):
    ranks = get_query_table_ranks(sample_info_dict, id_to_index, index_file, dim)
    mrr, p_scores = get_metrics(ranks)
    log_file = open(outfile, 'w', encoding='utf8')
    log_file.write('Eval Set' + '\t' + 'MRR')
    for p in find_p:
        log_file.write('\tP@' + str(p))
    log_file.write(eval_set + '\t' + str(mrr))
    for p in find_p:
        log_file.write('\t' + str(p_scores[p]))
    log_file.write('\n')
