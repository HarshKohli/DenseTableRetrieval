# Author: Harsh Kohli
# Date Created: 03-01-2022

import os

import keras.models
import yaml
import tensorflow as tf
from tqdm import tqdm
from transformers import TapasTokenizer, BertTokenizer
from models import TableEncoder
from utils.training_utils import TableDataset, get_dev_metrics
from utils.data_utils import load_pairwise_data, load_table_data, write_metrics

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = yaml.safe_load(open('config.yml', 'r'))
tf.config.run_functions_eagerly(config['execute_greedily'])

train_data = load_pairwise_data(config['pretrain_data'])
dev_data = load_pairwise_data(config['pretrain_dev_data'])
tables = load_table_data(config['filtered_tables'])

batch_size = config['batch_size']
model_name = config['model_name']

tokenizer = TapasTokenizer.from_pretrained(config['base_tokenizer'])
question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = TableEncoder(config)
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
save_path = os.path.join(config['model_dir'], model_name)
log_file = os.path.join(config['log_dir'], model_name + '.tsv')


@tf.function
def train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
               question_type):
    with tf.GradientTape() as tape:
        loss = model([input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                      question_type], training=True)
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


train_dataset = TableDataset(train_data, tables, tokenizer, question_tokenizer)
dev_data = TableDataset(dev_data, tables, tokenizer, question_tokenizer)

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
dev_dataloader = tf.data.Dataset.from_generator(dev_data, output_signature=output_signature).batch(batch_size)
num_iterations = str(int(len(train_data) // batch_size))

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
    model.save_weights(os.path.join(save_path, model_name) + '.h5')

    print('Done saving. Computing Dev scores...')
    dev_tables, dev_questions, dev_labels = [], [], []
    for dev_iteration, dev_batch in enumerate(dev_dataloader):
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = dev_batch
        table_embeddings = table_embedding_step(input_ids, attention_mask, token_type_ids)
        question_embeddings = question_embedding_step(question_inputs, question_mask, question_type)
        dev_tables.extend(table_embeddings.numpy())
        dev_questions.extend(question_embeddings.numpy())
        dev_labels.extend(labels.numpy())
    metrics_dict = get_dev_metrics(dev_tables, dev_questions, dev_labels)
    write_metrics(log_file, metrics_dict, epoch_num)
