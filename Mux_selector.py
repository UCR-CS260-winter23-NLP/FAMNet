#!/usr/bin/env python
# coding: utf-8

# In[1]:


# @author: Jayanta Banik, Ankit Gupta
# @tag: DarkSourceOfCode, halfdevilx333
# @afil: UCR grad student, UCR grad Student


# In[2]:


# Mux_selector using BERT


# In[3]:


# imports
import os
import datetime
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow_addons as tfa

from official.nlp import optimization

from datasets import load_dataset
from datasets import DownloadMode
from promptsource.templates import DatasetTemplates


# In[4]:


# experiment logger
name = "exp-000"
log_dir = "experiments"
#logger = SummaryWriter(log_dir=osp.join(log_dir, name))


# In[5]:


tf.get_logger().setLevel('ERROR')
strategy = tf.distribute.MirroredStrategy()
AUTOTUNE = tf.data.AUTOTUNE


# In[6]:


tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'


# In[7]:


epochs = 5
batch_size = 64
init_lr = 1e-5


# In[8]:


df = pd.read_csv("task.csv")


# In[9]:


df = df.sample(frac=.8)


# In[10]:


le = LabelEncoder()
df.w_task_set = le.fit_transform(df.w_task_set)


# In[11]:


X = tf.convert_to_tensor(df.input_text.values)
y = tf.one_hot(df.w_task_set, depth=len(le.classes_))


# In[12]:


num_examples, num_classes = y.shape


# In[13]:


in_memory_ds = {
    'answer' : y, 
    'question' : X 
}


# In[14]:


def make_bert_preprocess_model(sentence_features=['question'], seq_length=128):

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    truncated_segments = segments

    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


# In[15]:


def build_classifier_model(num_classes):

    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model


# In[16]:


def get_configuration():
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#     metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=4) 
    # metrics = tf.metrics.Accuracy()
    metrics = tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)
    return metrics, loss


# In[17]:


bert_preprocess_model = make_bert_preprocess_model()


# In[18]:


def load_dataset_from_tfds(in_memory_ds,  
                           batch_size,
                           bert_preprocess_model,
                           target='question'):
    
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds)
  
    num_examples = in_memory_ds[target].shape[0]

    dataset = dataset.shuffle(num_examples)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex['question']), ex['answer']))
    
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


# In[19]:


with strategy.scope():

    # metric have to be created inside the strategy scope
    metrics, loss = get_configuration()

    train_dataset, train_data_size = load_dataset_from_tfds(in_memory_ds,
                                                          batch_size, 
                                                          bert_preprocess_model,
                                                          target='answer')
    steps_per_epoch = train_data_size // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_train_steps // 10

    validation_dataset, validation_data_size = load_dataset_from_tfds(in_memory_ds,
                                                                    batch_size,
                                                                    bert_preprocess_model, 
                                                                    target='answer')
    validation_steps = validation_data_size // batch_size

    classifier_model = build_classifier_model(num_classes)

    optimizer = optimization.create_optimizer(
      init_lr=init_lr,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      optimizer_type='adamw')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])


# In[20]:


tfds_name = 'boolq'
sentence_features = ['question']


# In[27]:


main_save_path = 'my_models'
bert_type = tfhub_handle_encoder.split('/')[-2]
saved_model_name = f'{tfds_name.replace("/", "_")}_{bert_type}'

saved_model_path = os.path.join(main_save_path, saved_model_name)

preprocess_inputs = bert_preprocess_model.inputs
bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
bert_outputs = classifier_model(bert_encoder_inputs)
model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

# with tf.device('/job:localhost'):
reloaded_model = tf.saved_model.load(saved_model_path)


# In[21]:


def prepare(record):
    model_inputs = [[record[ft]] for ft in sentence_features]
    return model_inputs


# In[22]:


def prepare_serving(record):
    model_inputs = {ft: record[ft] for ft in sentence_features}
    return model_inputs


# In[23]:


def print_bert_results(test, bert_result, dataset_name):

    bert_result_class = tf.argmax(bert_result, axis=1)[0]
    print('sentence:', test[0].numpy())
    
    print('BERT results:', le.classes_[bert_result_class])

    print('BERT raw results:', bert_result[0])
    print()


# In[72]:


def predict(s):
    
    s = tf.convert_to_tensor([s.encode('utf-8')], dtype='string', name='question')
    result = reloaded_model(s)
    bert_result_class = tf.argmax(result, axis=1)[0]
    return le.classes_[bert_result_class]


# In[73]:


# predict('True or False? Sun rises from east?')

