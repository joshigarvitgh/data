#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bert-tensorflow')
get_ipython().system('pip install tensorflow_hub')
get_ipython().system('pip install pytorch_pretrained_bert')
#!pip install transformers


# In[1]:


import pandas as pd
import re
import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertConfig,BertForPreTraining
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
#import logging
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#from transformers import *


# In[2]:


data=pd.read_csv('freqfilter.csv')


# In[3]:


print(data.head())


# In[4]:


print(data['TEXT'][1])
dischargeset=data['TEXT']


# In[5]:


dischargeset_clean= data['TEXT'].map(lambda x: re.sub('[!@#$*]', '', x))
dischargeset_clean[1]


# In[6]:


#dischargeset.to_csv('file.csv', sep='\t')


# # BERT Normal

# In[ ]:





# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

# import bert
# from bert import run_classifier
# from bert import optimization
# from bert import tokenization

# In[7]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[8]:


data1=dischargeset_clean[:1000]


# In[9]:


marked_text = "[CLS] " + data1[0] + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexT = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_text)


# for tup in zip(G, indexT):
#     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

# In[10]:


#we are making the sentence pairing here ie. sentence 0 or sentence 1
segments_ids =[1]*len(tokenized_text)
print (segments_ids)


# In[11]:


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexT])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-cased')


# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


# In[12]:


with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)


# In[13]:


print ("Number of layers:", len(encoded_layers))
layer_i = 0

print ("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))


# In[14]:


# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = encoded_layers[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()


# In[15]:


# `encoded_layers` is a Python list.
print('     Type of encoded_layers: ', type(encoded_layers))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', encoded_layers[0].size())


# In[16]:


# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(encoded_layers, dim=0)

token_embeddings.size()


# In[17]:


# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings.size()


# In[18]:


# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()


# In[19]:


#word vector summing last 4 layers

# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)
    
    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


# In[20]:


#senteve vector
# `encoded_layers` has shape [12 x 1 x 22 x 768]

# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = encoded_layers[11][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)

print ("Our final sentence embedding vector of shape:", sentence_embedding.size())


# In[21]:


for i, token_str in enumerate(tokenized_text):
  print (i, token_str)


# In[22]:


from scipy.spatial.distance import cosine

# Calculate the cosine similarity between the words
#(different meanings).
diff_word = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])

# Calculate the cosine similarity between the word bank
# in "bank robber" vs "bank vault" (same meaning).
same_word = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_word)
print('Vector similarity for *different* meanings:  %.2f' % diff_word)


# #now for icd code prediction
