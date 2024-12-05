# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-24 11:48:59
 LastEditTime : 2023-08-24 19:42:16
 Copyright (C) 2023 mryxj. All rights reserved.
'''

from calflops import calculate_flops
from calflops.utils import get_module_flops
from transformers import AutoModel
from transformers import AutoTokenizer
from collections import deque
from functools import partial

def store_flops(module, datastore=None):
    data = dict()
    children = list(module.named_children())
    if len(children) > 0:
        datastore.rotate(len(children))
        for name, child in children:
            typename, flops, child_data = datastore.popleft()
            data[name] = (typename, flops, child_data,)
    datastore.append((type(module).__name__, get_module_flops(module), data,))

myqueue = deque()

batch_size = 1
max_seq_length = 128
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

flops, macs, params = calculate_flops(model=model,
                                      extra_apply_funcs=[partial(store_flops, datastore=myqueue)],
                                      print_detailed=False,
                                      input_shape=(batch_size,max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
print(myqueue)
