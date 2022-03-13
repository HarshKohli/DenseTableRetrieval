# Author: Harsh Kohli
# Date Created: 06-02-2022

import os
import yaml
import tensorflow as tf
from tqdm import tqdm
from transformers import TapasTokenizer, BertTokenizer
from models import TableEncoder
from utils.training_utils import PairwiseDataset, get_dev_metrics
from utils.data_utils import load_pairwise_data, load_table_data, write_metrics, create_dir_if_not_exists

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

train_data = load_pairwise_data(config['finetune_train_preprocessed'])
dev_data = load_pairwise_data(config['finetune_dev_preprocessed'])
test_data = load_pairwise_data(config['finetune_test_preprocessed'])

tables = load_table_data(config['nq_tables_preprocessed'])

batch_size = config['batch_size']
model_name = config['model_name']

tokenizer = TapasTokenizer.from_pretrained(config['base_tokenizer'])
question_tokenizer = BertTokenizer.from_pretrained(config['base_question_encoder'])

model = TableEncoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], model_name)
log_file_dev = os.path.join(config['log_dir'], model_name + '_dev.tsv')
log_file_test = os.path.join(config['log_dir'], model_name + '_test.tsv')
create_dir_if_not_exists(save_path)


@tf.function
def train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
               question_type):
    with tf.GradientTape() as tape:
        loss = model([input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                      question_type], training=True)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        (grad, var)
        for (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None
    )
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
dev_dataset = PairwiseDataset(dev_data, tables, tokenizer, question_tokenizer)
test_dataset = PairwiseDataset(test_data, tables, tokenizer, question_tokenizer)

output_signature = (
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32))

train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).shuffle(
    1000).batch(batch_size)
dev_dataloader = tf.data.Dataset.from_generator(dev_dataset, output_signature=output_signature).batch(batch_size)
test_dataloader = tf.data.Dataset.from_generator(test_dataset, output_signature=output_signature).batch(batch_size)
num_iterations = str(int(len(train_data) // batch_size))

if config['load_weights']:
    print('Loading pretrained weights...')
    input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = next(
        iter(train_dataloader))
    loss = train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                      question_type)
    model.load_weights(config['pretrained_model'])
    print('Done Loading pretrained weights...')

print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    iteration = 0
    for batch in tqdm(train_dataloader):
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = batch
        loss = train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                          question_type)
        if iteration % 100 == 0:
            print('Done with ' + str(iteration) + ' iterations out of ' + num_iterations + '. Loss is ' + str(
                loss.numpy()))
        iteration = iteration + 1

    print('Completed Epoch. Saving Latest Model...')
    model.save_weights(os.path.join(save_path, str(epoch_num)) + '.h5')

    print('Done saving. Computing Dev scores...')
    dev_tables, dev_questions, dev_labels = [], [], []
    for dev_iteration, dev_batch in enumerate(dev_dataloader):
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = dev_batch
        table_embeddings = table_embedding_step(input_ids, attention_mask, token_type_ids)
        question_embeddings = question_embedding_step(question_inputs, question_mask, question_type)
        dev_tables.extend(table_embeddings.numpy())
        dev_questions.extend(question_embeddings.numpy())
        dev_labels.extend(labels.numpy())

    print('Computing Test scores...')
    test_tables, test_questions, test_labels = [], [], []
    for test_iteration, test_batch in enumerate(test_dataloader):
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = test_batch
        table_embeddings = table_embedding_step(input_ids, attention_mask, token_type_ids)
        question_embeddings = question_embedding_step(question_inputs, question_mask, question_type)
        test_tables.extend(table_embeddings.numpy())
        test_questions.extend(question_embeddings.numpy())
        test_labels.extend(labels.numpy())

    metrics_dict = get_dev_metrics(dev_tables, dev_questions, dev_labels)
    write_metrics(log_file_dev, metrics_dict, epoch_num)

    metrics_dict = get_dev_metrics(test_tables, test_questions, test_labels)
    write_metrics(log_file_test, metrics_dict, epoch_num)
