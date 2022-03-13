# Author: Harsh Kohli
# Date Created: 03-01-2022

import os
import yaml
import tensorflow as tf
from tqdm import tqdm
from transformers import TapasTokenizer, BertTokenizer
from models import TableEncoder
from utils.training_utils import PairwiseDataset, get_dev_metrics
from utils.data_utils import load_pairwise_data, load_table_data, write_metrics, create_dir_if_not_exists

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

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
question_tokenizer = BertTokenizer.from_pretrained(config['base_question_encoder'])

save_path = os.path.join(config['model_dir'], model_name)
log_file = os.path.join(config['log_dir'], model_name + '.tsv')
create_dir_if_not_exists(save_path)

train_dataset = PairwiseDataset(train_data, tables, tokenizer, question_tokenizer)
dev_dataset = PairwiseDataset(dev_data, tables, tokenizer, question_tokenizer)

output_signature = (
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32),
    tf.TensorSpec(shape=(128), dtype=tf.int32))

with strategy.scope():
    train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).shuffle(
        1000).batch(batch_size)
    train_dataloader = train_dataloader.with_options(options)
    dist_train_dataloader = strategy.experimental_distribute_dataset(train_dataloader)

    dev_dataloader = tf.data.Dataset.from_generator(dev_dataset, output_signature=output_signature).batch(batch_size)
    dev_dataloader = dev_dataloader.with_options(options)
    dist_dev_dataloader = strategy.experimental_distribute_dataset(dev_dataloader)

    model = TableEncoder(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])


def train_step(train_batch):
    with tf.GradientTape() as tape:
        per_example_loss = model(train_batch, training=True)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        (grad, var)
        for (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None
    )
    return loss


def table_embedding_step(batch):
    input_ids, attention_mask, token_type_ids = batch
    table_embeddings = model.get_table_embedding(input_ids, attention_mask, token_type_ids)
    return table_embeddings


def question_embedding_step(batch):
    question_inputs, question_mask, question_type = batch
    question_embeddings = model.get_question_embedding(question_inputs, question_mask, question_type)
    return question_embeddings


@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    losses = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return losses


@tf.function
def distributed_table_embedding_step(dist_inputs):
    table_embeddings = strategy.run(table_embedding_step, args=(dist_inputs,))
    return table_embeddings


@tf.function
def distributed_question_embedding_step(dist_inputs):
    question_embeddings = strategy.run(question_embedding_step, args=(dist_inputs,))
    return question_embeddings


num_iterations = str(int(len(train_data) // batch_size))
print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    iteration = 0
    for batch in tqdm(dist_train_dataloader):
        loss = distributed_train_step(batch)
        if iteration % 100 == 0:
            print('Done with ' + str(iteration) + ' iterations out of ' + num_iterations + '. Loss is ' + str(
                loss.numpy()))
        iteration = iteration + 1
        break

    print('Completed Epoch. Saving Latest Model...')
    model.save_weights(os.path.join(save_path, str(epoch_num)) + '.h5')

    print('Done saving. Computing Dev scores...')
    dev_tables, dev_questions, dev_labels = [], [], []
    for dev_iteration, dev_batch in enumerate(dev_dataloader):
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = dev_batch
        table_embeddings = distributed_table_embedding_step((input_ids, attention_mask, token_type_ids))
        question_embeddings = distributed_question_embedding_step((question_inputs, question_mask, question_type))
        dev_tables.extend(table_embeddings.numpy())
        dev_questions.extend(question_embeddings.numpy())
        dev_labels.extend(labels.numpy())

    metrics_dict = get_dev_metrics(dev_tables, dev_questions, dev_labels)
    write_metrics(log_file, metrics_dict, epoch_num)
