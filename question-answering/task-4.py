#!/usr/bin/env python
# coding: utf-8

# In[34]:


from google.colab import drive
drive.mount('./gdrive')


# In[5]:


get_ipython().system(' pip install razdel')
get_ipython().system(' pip install tqdm')
get_ipython().system(' pip install nltk')


# In[6]:


import csv
import string
from collections import defaultdict
from tqdm import tnrange, tqdm_notebook
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import re
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
from razdel import sentenize
from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine


# In[8]:


SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)


# In[7]:


# Russian NLP utilites
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stemmer = nltk.stem.snowball.SnowballStemmer("russian")


# In[9]:


DEVICE = torch.device('cuda' 
                      if torch.cuda.is_available() 
                      else 'cpu')
DEVICE


# In[15]:


def find_closest_sent(text, question, incl_next):
  sents = [elem.text for elem in list(sentenize(text))]
  #question_tokens = [stemmer.stem(word.lower()) for word in question.split(" ")]
  docs = sents + [question]
  #print(docs)
  modified_docs = [[stemmer.stem(i.lower()) for i in tokenize(d.translate(str.maketrans('','',string.punctuation))) if i.lower() not in stop_words] for d in docs]
  #print(modified_docs)
  modified_docs = [' '.join(i) for i in modified_docs] # this is only to convert our list of lists to list of strings that vectorizer uses.
  tf_idf = TfidfVectorizer().fit_transform(modified_docs)

  l = len(sents)
  minimum = (1, None)
  for i in range(l):
    #print(sents[i])
    #print(cosine(tf_idf[i].todense(), tf_idf[l].todense()))
    if cosine(tf_idf[i].todense(), tf_idf[l].todense()) < minimum[0]:
      minimum = (cosine(tf_idf[i].todense(), tf_idf[l].todense()), i)
  if minimum[1] == None:
    return None, -1
  #if minimum[1] + 1 != l and incl_next:
  #  return sents[minimum[1]] + " " + sents[minimum[1] + 1], minimum[1]
  return sents[minimum[1]], minimum[1]


# In[16]:


def find_sent_and_ans_start(text, question, answer):
  closest_sent, closest_sent_ind = find_closest_sent(text, question, False)
  #print(closest_sent)

  #sent_text = nltk.sent_tokenize(text, language="russian")
  sent_text = [elem.text for elem in list(sentenize(text))]
  j = 0
  for sent in sent_text:
    ind = sent.lower().find(answer.lower())
    if ind != -1:
      if closest_sent != None and closest_sent.strip() != sent.strip():
        if closest_sent_ind < j:
          sent = closest_sent + " " + sent
          ind += len(closest_sent) + 1
        else:
          sent = sent + " " + closest_sent
      return sent, ind
    j += 1

  i = 0
  sz = len(sent_text)
  for i in range(sz - 1):
    sent = sent_text[i] + " " + sent_text[i + 1]
    #print(sent)
    ind = sent.lower().find(answer.lower())
    if ind != -1:
      if closest_sent != None and sent.strip().find(closest_sent.strip()) == -1:
        if closest_sent_ind < i:
          sent = closest_sent + " " + sent
          ind += len(closest_sent) + 1
        else:
          sent = sent + " " + closest_sent
      return sent, ind
  
  return "", -1


# In[17]:


text = "В некоторые жаркие и засушливые годы обнаруживается связь между погодными условиями и изменениями внешнего вида бабочек: появление бабочек с признаками южных климатических форм. Смена времён года вызывает у многих видов явления сезонного диморфизма и триморфизма. Среди дневных бабочек можно также найти также несколько примеров полиморфизма — одновременного и совместного существования различно окрашенных форм в одном виде, свободно скрещивающихся между собою и передающих свои признаки потомству (например, желтушки, большая перламутровка и другие)."
question = "Какие представители вида бабочек обладают полиморфизмом"
ans = "желтушки, большая перламутровка"
sent, ans_start = find_sent_and_ans_start(text, question, ans)
sent


# In[18]:


sentens = nltk.sent_tokenize("Через два года после смерти Л. Пастера в 1897 году Э. Бухнер опубликовал работу Спиртовое брожение без дрожжевых клеток , в которой экспериментально показал, что бесклеточный дрожжевой сок осуществляет спиртовое брожение так же, как и неразрушенные дрожжевые клетки. В 1907 году за эту работу он был удостоен Нобелевской премии. Впервые высокоочищенный кристаллический фермент (уреаза) был выделен в 1926 году Дж. Самнером. В течение последующих 10 лет было выделено ещё несколько ферментов, и белковая природа ферментов была окончательно доказана.", language="russian")
sentens


# In[23]:


df = pd.read_csv("train_qa.csv", encoding="utf-8")
size = df.shape[0]

df.head(5)


# In[24]:


contexts = df.iloc[:, 2].values
questions = df.iloc[:, 3].values
answers = df.iloc[:, 4].values

answers[34157] = "573,000 копий"


# In[25]:


def split_text(text):
  return re.split('(\W)', text)


# In[26]:


skipped = 0
dataset = []

for i in range(size):
  context = contexts[i]
  question = questions[i].strip("?")
  answer = answers[i].strip(".").strip("...").strip().strip("?")

  sent, ans_start = find_sent_and_ans_start(context, question, answer)
  ans_len = len(answer)

  if i % 5000 == 0:
    print(i)
    print("CONTEXT: ", context)
    print("QUESTION: ", question)
    print("ANSWER: ", answer)
    if ans_start != -1:
      print("EXTRACTED ANSWER: ", sent[ans_start:ans_start + ans_len])
    print("SENT: ", sent)

  if ans_start != -1:
      elem = {'context': split_text(sent.lower()), 'question': split_text(question.lower()), 'answer': (ans_start, ans_len)}
      dataset.append(elem)
  else:
      print(i)
      print(context)
      print(answer)
      skipped += 1
  #ind = context.find(answer)

  #start_ind[i] = ind
  #end_ind[i] = ind + len(answer)

print("Skipped:", skipped)


# In[27]:


print("Skipped:", skipped)


# In[28]:


text = "Во 2-й половине 1960-х годов Фишер выдвинулся в число сильнейших шахматистов мира, добиваясь успехов в турнирах самого высокого ранга: Гавана (1965) — 2—4-е место (в этом турнире Фишер участвовал заочно — в 1960-х годах США ввели санкции против Кубы, госдепартамент не разрешил ему выезд в Гавану, и Роберт играл из США, передавая свои ходы по телефону[5]); Санта-Моника (1966) — 2-е; Охрид и Монте-Карло (1967) — 1-е; Нетанья и Винковци (1968) — 1-е; Ровинь — Загреб и Буэнос-Айрес (1970) — 1-е место."
sent_text = [elem.text for elem in list(sentenize(text))]
print(sent_text)


# In[29]:


print(len(dataset))


# In[30]:


print(dataset[0])


# In[83]:


sep_token = "<SEP>"
pad_token = "<PAD>"
word2ind = defaultdict(lambda: 0)
word2ind[sep_token] = 1
word2ind[pad_token] = 2


# In[84]:


def add_entity(item, entity, mode):
  seq = []
  for token in item[entity]:
    st = stemmer.stem(token)
    if mode == "train" and st not in word2ind:
      word2ind[st] = len(word2ind) + 1
    seq.append(word2ind[st])
  if entity == "context":
    seq.append(word2ind[sep_token])
  return seq

def to_seq(item, mode):
  seq = []
  seq += add_entity(item, "context", mode)
  seq += add_entity(item, "question", mode)
  return torch.tensor(seq)

def pad(dataset, mode):
    seqs = [to_seq(item, mode) for item in tqdm(dataset)]
    return pad_sequence(seqs, batch_first=True, padding_value=word2ind[pad_token])


# In[85]:


padded_dataset = pad(dataset, "train")


# In[116]:


padded_dataset.shape


# In[86]:


print(len(word2ind))


# In[87]:


def to_words(item):
    pos = []
    i = 0
    for word in item['context']:
        word_len = len(word)
        pos.append((i, word_len))
        i += word_len
    return pos

def to_words_pos(dataset):
    return [to_words(item) for item in tqdm(dataset)]


# In[88]:


to_words_pos_dataset = to_words_pos(dataset)


# In[89]:


def merge(padded_dataset, to_words_pos_dataset, dataset):
    merged = []
    for sent, word_pos, row in tqdm(zip(padded_dataset, to_words_pos_dataset, dataset)):
        ans_start, ans_len = row['answer']
        ans_end = ans_start + ans_len
        y = np.zeros((2), dtype=int)
        for i in range(len(word_pos)):
            if word_pos[i][0] == ans_start:
                y[0] = i
            if word_pos[i][0] == ans_end:
                y[1] = i - 1
        merged.append(np.append(sent.numpy(), y))
    return merged


# In[90]:


merged_dataset = merge(padded_dataset, to_words_pos_dataset, dataset)


# In[92]:


class BiLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, vocab_size):
        super(BiLSTM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 2)
    
    def forward(self, x):
        #print(x.shape) torch.Size([64, 2727])
        x = self.word_embeddings(x)
        #print(x.shape) torch.Size([64, 2727, 64])
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        #print(x.shape) torch.Size([64, 2, 2727])
        x = F.log_softmax(x, dim=2)
        # print(x.shape) torch.Size([64, 2, 2727])
        
        return x


# In[94]:


NUM_EPOCHS = 10
BATCH_SIZE = 64
EMBEDDING = 64
HIDDEN = 64
NUM_LAYERS = 3


# In[95]:


torch.cuda.empty_cache()


# In[96]:


model = BiLSTM(EMBEDDING, HIDDEN, NUM_LAYERS, (len(word2ind)) + 1)
model = model.float().to(DEVICE)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[97]:


train_dataset, val_dataset = train_test_split(merged_dataset, test_size=0.2, random_state=42)


# In[98]:


print(train_dataset[1].shape)


# In[99]:


train_losses = []
train_x = []
val_losses = []
val_x = []

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

with torch.no_grad():
    losses = []
    for data in val_loader:
        split_pos = data.shape[1] - 2
        x = data[:, :split_pos].to(DEVICE)
        y = data[:, split_pos:].to(DEVICE)
        output = model(x.long())
        y1 = y[:, 0].reshape(-1)
        y2 = y[:, 1].reshape(-1)
        loss = (loss_func(output[:, 0], y1.long()) + 
                loss_func(output[:, 1], y2.long())) / 2
        losses.append(loss.item())

    val_x.append(0)
    val_losses.append(np.mean(np.array(losses)))

it = 1

for epoch in tqdm.notebook.trange(NUM_EPOCHS, desc='Epoch'):
    for data in train_loader:
        split_pos = data.shape[1] - 2
        x = data[:, :split_pos].to(DEVICE)
        y = data[:, split_pos:].to(DEVICE)
        optimizer.zero_grad()
        output = model(x.long())
        y1 = y[:, 0].reshape(-1)
        y2 = y[:, 1].reshape(-1)
        loss = (loss_func(output[:, 0], y1.long()) + 
                loss_func(output[:, 1], y2.long())) / 2
        loss.backward()

        train_x.append(it)
        train_losses.append(loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        it += 1

    with torch.no_grad():
        losses = []
        for data in val_loader:
            split_pos = data.shape[1] - 2
            x = data[:, :split_pos].to(DEVICE)
            y = data[:, split_pos:].to(DEVICE)
            output = model(x.long())
            y1 = y[:, 0].reshape(-1)
            y2 = y[:, 1].reshape(-1)
            loss = (loss_func(output[:, 0], y1.long()) + 
                loss_func(output[:, 1], y2.long())) / 2
            losses.append(loss.item())

        val_x.append(it)
        val_losses.append(np.mean(np.array(losses)))


# In[100]:


plt.plot(train_x, train_losses, label='train')
plt.plot(val_x, val_losses, label='val')
plt.legend()
plt.title("loss")
plt.show()


# In[133]:


max_len = 0

test_df = pd.read_csv("dataset_281937_1.txt", encoding="utf-8", delimiter='\t')
test_size = test_df.shape[0]

test_dataset = []

test_question_ids = test_df.iloc[:, 1].values
test_contexts = test_df.iloc[:, 2].values
test_questions = test_df.iloc[:, 3].values

for i in range(test_size):
  context = test_contexts[i]
  question = test_questions[i].strip("?")
  question_id = test_question_ids[i]

  sent, _ = find_closest_sent(context, question, True)

  if sent == None:
    #print(context)
    #print(question)
    sent = context
    #print(split_text(context))

  max_len = max(max_len, len(split_text(context)) + len(split_text(question)))

  #if max_len == 600:
  #  print(context)
  #  print(question)
  #  print(split_text(context))

  test_dataset.append({'context': split_text(sent.lower()),
            'question': split_text(question.lower()),
            'question_id': question_id})

padded_test = pad(test_dataset, "test")


# In[131]:


print(padded_test.shape)
print(max_len)


# In[115]:


padded_test


# In[132]:


with torch.no_grad():
  test_loader = torch.utils.data.DataLoader(padded_test, batch_size=BATCH_SIZE)
  ans = None

  for data in test_loader:
    x = data.to(DEVICE)
    output = model(x.long())
    value, ansx = output.max(dim=2)
    ansx = ansx.cpu().numpy()
    if ans is None:
      ans = ansx
      continue
    ans = np.append(ans, ansx, axis=0)

out_file = open("./out.txt", "w")
for positions, item in zip(ans, test_dataset):
    start, end = positions
    if start > end:
        start, end = end, start
    if end >= len(item['context']):
        start, end = 0, len(item['context']) - 1
    #print(type(item['question_id']))
    #print(type("".join(item['context'][start:end + 1])))
    out_file.write(str(item['question_id']) + "\t" + "".join(item['context'][start:end + 1]) + "\n")

out_file.close()


# In[ ]:





# In[ ]:




