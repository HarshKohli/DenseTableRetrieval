# Author: Harsh Kohli
# Date Created: 19-02-2022


import yaml
import tensorflow as tf
from transformers import TapasTokenizer, BertTokenizer
from models import TableEncoder
from utils.training_utils import PairwiseDataset, TablesDataset
from utils.data_utils import load_pairwise_data, load_table_data, filter_positive_testset

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

train_data = load_pairwise_data(config['finetune_train_preprocessed'])
test_data = load_pairwise_data(config['finetune_test_preprocessed'])
tables = load_table_data(config['nq_tables_preprocessed'])

test_data = filter_positive_testset(test_data)

batch_size = config['batch_size']
model_name = config['model_name']

tokenizer = TapasTokenizer.from_pretrained(config['base_tokenizer'])
question_tokenizer = BertTokenizer.from_pretrained(config['base_question_encoder'])

model = TableEncoder(config)


@tf.function
def train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
               question_type):
    loss = model([input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                  question_type], training=True)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def table_embedding_step(input_ids, attention_mask, token_type_ids):
    table_embeddings = model.get_table_embedding(input_ids, attention_mask, token_type_ids)
    return table_embeddings


@tf.function
def question_embedding_step(question_inputs, question_mask, question_type):
    question_embeddings = model.get_question_embedding(question_inputs, question_mask, question_type)
    return question_embeddings


train_dataset = PairwiseDataset(train_data, tables, tokenizer, question_tokenizer)
test_dataset = PairwiseDataset(test_data, tables, tokenizer, question_tokenizer)
tables_dataset = TablesDataset(tables, tokenizer)

output_signature = (
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32))

tables_output_signature = (
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.string))

train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).batch(batch_size)
test_dataloader = tf.data.Dataset.from_generator(test_dataset, output_signature=output_signature).batch(batch_size)
tables_dataloader = tf.data.Dataset.from_generator(tables_dataset, output_signature=tables_output_signature).batch(
    batch_size)

print('Loading model...')
input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = next(
    iter(train_dataloader))
loss = train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                  question_type)
model.load_weights(config['pretrained_model'])
print('Done Loading model... Computing Question Embeddings')

num_iterations = str(int(len(test_data) // batch_size))
test_questions, test_labels = [], []
for test_iteration, test_batch in enumerate(test_dataloader):
    input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = test_batch
    question_embeddings = question_embedding_step(question_inputs, question_mask, question_type)
    test_questions.extend(question_embeddings.numpy())
    test_labels.extend(labels.numpy())
    if test_iteration % 100 == 0:
        print('Done with ' + str(test_iteration) + ' iterations out of ' + num_iterations)

print('Computing Table Embeddings... ')
num_iterations = str(int(len(tables) // batch_size))
test_tables = {}
for test_iteration, test_batch in enumerate(tables_dataloader):
    input_ids, attention_mask, token_type_ids, table_ids = test_batch
    table_embeddings = table_embedding_step(input_ids, attention_mask, token_type_ids)
    for table_id, table_embedding in zip(table_ids, table_embeddings):
        test_tables[table_id.numpy().decode('ascii')] = table_embedding
    if test_iteration % 100 == 0:
        print('Done with ' + str(test_iteration) + ' iterations out of ' + num_iterations)
