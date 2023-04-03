import numpy as np
import pandas as pd
import json

import gensim.downloader as api
from sklearn.manifold import TSNE

with open('fake_news.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Load pre-trained word embeddings from gensim library
model = api.load('glove-twitter-25')

# Remove rows with missing embeddings
df = df[df['text'].apply(lambda x: any(word in model for word in x.split()))]

# Convert text data to embeddings representation:
X = np.array([np.mean([model[word] for word in doc.split() if word in model], axis=0) for doc in df['text']])

# Convert each document to a vector of word embeddings
X = []
for doc in df['text']:
    doc_vec = [model[word] for word in doc.split() if word in model]
    if len(doc_vec) > 0:
        X.append(np.mean(doc_vec, axis=0))
X = np.array(X)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['is_deceptive'])
plt.show()