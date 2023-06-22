import glob
import pickle
import numpy as np
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors
from tqdm import tqdm

save_path = "chat_data/"
MODEL_FILE = "enwiki_20180420_100d.txt"

maxlen = 40

files = glob.glob("*.p")
X = ["today i received an insane nft sponsor offer and the amounts are astronomical one"]
Y = ["BatChest"]
print(files)
for file in files:
    with open(file, "rb") as f:
        samples = pickle.load(f)
        for sample in samples:
            X.append(sample[0])
            Y.append(sample[1])

counts = dict()
for sample in Y:
    counts[sample] = counts.get(sample, 0) + 1

print(counts)

emote_file = "emotes.txt"

with open(emote_file, 'r') as f:
    emotes = f.read().split(',')

with open("test.csv", 'r') as f:
    test_lines = f.read().split('\n')

X_test = []
y_test = []
for line in test_lines:
    X_test.append(line.split(',')[0])
    y_test.append(line.split(',')[1])


def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict


def sparse_sector(X, Y):
    xs = []
    ys = []
    # convert emotes to sparse vector
    for x, y in zip(X, Y):
        if y in emotes:
            zeros = np.zeros(len(emotes))
            zeros[emotes.index(y)] = 1
            ys.append(zeros)
            xs.append(x.split(' '))
    return xs, ys


def get_embedding_matrix(fname, unique_words):
    word_vectors = KeyedVectors.load_word2vec_format(fname, binary=False)
    keys = set(word_vectors.index_to_key)
    unseen_vec = 0
    embedding_matrix = np.zeros((vocab_size, 100))
    for i, word in tqdm(enumerate(unique_words), total=len(unique_words), desc="creating word embedding"):
        if i > 2:
            embedding_vector = (word_vectors.get_vector(word) if word in keys
                                else unseen_vec)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


xs, ys = sparse_sector(X, Y)
X_test, y_test = sparse_sector(X_test, y_test)

vocab = build_token_dict(xs + X_test)
vocab_size = len(vocab)
unique_words = list(vocab.keys())
embedding_matrix = get_embedding_matrix(MODEL_FILE, unique_words)

with open(save_path + "embedding_matrix.p", "wb") as f:
    pickle.dump(embedding_matrix, f)


def fiturize(xs, vocab, maxlen):
    # Padding
    encode_tokens = [tokens + ['<PAD>'] * (maxlen - len(tokens)) for tokens in xs]
    samples = [list(map(lambda x: vocab[x], tokens)) for tokens in encode_tokens]
    return samples


xs = fiturize(xs, vocab, maxlen)
X_test = fiturize(X_test, vocab, maxlen)
data_size = len(xs + X_test)

xs = np.array(xs)
ys = np.array(ys)
X_test = np.array(X_test)
y_test = np.array(y_test)
# cross validation
kf = KFold(n_splits=4, random_state=420, shuffle=True)
fold = 0
for train_index, val_index in kf.split(xs, ys):
    X_train, y_train = xs[train_index], ys[train_index]
    X_val, y_val = xs[val_index], ys[val_index]
    print("Data size: ", data_size)
    print("Vocabulary size: ", vocab_size)
    print("Max sequence length: ", maxlen)
    print(len(X_train), "Training sequences")
    print(len(X_val), "Validation sequences")
    print(len(X_test), "Test sequences")
    with open("{}train_{}.p".format(save_path, str(fold)), "wb") as f:
        pickle.dump([X_train, y_train, X_val, y_val, X_test, y_test], f)
    fold += 1
