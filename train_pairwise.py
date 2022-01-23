# Author: Harsh Kohli
# Date Created: 03-01-2022

import os
import yaml
import tensorflow as tf
from transformers import TapasTokenizer, BertTokenizer
from models import TableEncoder
from utils.training_utils import TableDataset
from utils.data_utils import load_pairwise_data, load_table_data

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
log_path = os.path.join(config['log_dir'], model_name)


@tf.function
def train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
               question_type):
    with tf.GradientTape() as tape:
        loss = model([input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                      question_type])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        (grad, var)
        for (grad, var) in zip(gradients, model.trainable_variables)
        if grad is not None
    )
    return loss


train_dataset = TableDataset(train_data, tables, tokenizer, question_tokenizer)
output_signature = (
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512,), dtype=tf.int32),
    tf.TensorSpec(shape=(512, 7), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(512), dtype=tf.int32),
    tf.TensorSpec(shape=(512), dtype=tf.int32),
    tf.TensorSpec(shape=(512), dtype=tf.int32))
train_dataloader = tf.data.Dataset.from_generator(train_dataset, output_signature=output_signature).batch(batch_size)

print('Starting Training...')
for epoch_num in range(config['num_epochs']):
    print('Starting Epoch: ' + str(epoch_num))
    iteration = 0
    for batch in train_dataloader:
        input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask, question_type = batch
        loss = train_step(input_ids, attention_mask, token_type_ids, questions, labels, question_inputs, question_mask,
                          question_type)
        if iteration % 10 == 0:
            print('Done with ' + str(iteration) + ' iterations. Loss is ' + str(loss.numpy()))
        iteration = iteration + 1

    print('Completed Epoch. Saving Latest Model...')
    tf.keras.models.save_model(model, os.path.join(save_path, str(epoch_num)))
