#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import torch
import numpy as np
from tqdm import tqdm
from simpletransformers.t5 import T5Model
from datasets import load_dataset, DownloadMode, load_metric
from transformers import T5Tokenizer
from transformers import pipeline


# In[8]:


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 64,
    "train_batch_size": 64,
    "num_train_epochs": 2,
    "save_eval_checkpoints": True,
    "save_steps": -1,
    "use_multiprocessing": False,
    "fp16": True,
    "wandb_project": "T5-BART",
}

model = T5Model("t5", 'HoSoTAs/en-de', args=model_args)


# In[ ]:


def get_summarized_string(s):
    return model.predict([s])

