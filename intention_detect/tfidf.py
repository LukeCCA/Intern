import jieba
import numpy as np
import pandas as pd
from ckiptagger import WS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, required=True, help="select dataset")
parser.add_argument("-ws_mode", type=int, required=True, choices=[0,1,2,3], help="select word segmenter;\n 0 for jeiba, 1 for ckip,\n 2 for ckip->jeiba, 3 for jeiba->ckip")
args = parser.parse_args()

DATASET = args.dataset
WS_SET = ["jieba", "ckip", "mix_0", "mix_1"]
WORD_SEGMENTER = WS_SET[args.ws_mode]

# function definition
def get_tfidf(corpus, voc, idf):
    count_vectorizer = CountVectorizer(vocabulary=voc, token_pattern=r"(?u)\b\w+\b")
    tf = count_vectorizer.fit_transform(corpus)
    tfidf = (np.ma.log(tf.toarray()) + 1) * idf
    tfidf = tfidf.filled(0)
    
    return normalize(tfidf, norm='l2')

def get_similarity(test_tfidf, train_tfidf, num_labels=None):
    if num_labels is None:
        sim_array = cosine_similarity(test_tfidf, train_tfidf)
    else:
        train_tfidf_by_class = []
        start_idx = 0
        for i in num_labels:
            train_tfidf_by_class += [train_tfidf[start_idx : start_idx+i].mean(axis=0)]
            start_idx += i
        train_tfidf_by_class = np.stack(train_tfidf_by_class)
        sim_array = cosine_similarity(test_tfidf, train_tfidf_by_class)
    
    return np.array(sim_array)

def get_prediction(sim_array, test_data, train_data, sim_mode):
    correct = 0
    for i, ans in enumerate(test_data["label"]):
        if sim_mode == 0:
            sorted_arg = np.argsort(sim_array[i])
            sorted_arg = np.flip(sorted_arg)[:10] # 10 for top 10 similar
            sorted_pred = [train_data["label"][idx] for idx in sorted_arg]
            pred = max(set(sorted_pred), key=sorted_pred.count)
        else:
            pred = np.argmax(sim_array[i])
    
        if pred == ans:
            correct += 1
    acc = correct / len(test_data["label"])
    return acc

if WORD_SEGMENTER == "ckip" or WORD_SEGMENTER.find("mix") != -1:
    ws = WS("../ckiptagger/data")

def get_model_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.sort_values(by=["labels"], ignore_index=True)
    src_labels = sorted(set(df.labels.tolist()))
    df["labels"] = [src_labels.index(l) for l in df.labels.tolist()]
    texts = df["texts"]
    labels = df["labels"]
    num_labels = list(Counter(labels).values())
    
    data = {"corpus":[], "label":[], "src_texts":[], "src_label":[]}
    
    for i, t in (enumerate(texts)):
        label = labels[i]
        
        if WORD_SEGMENTER == "ckip":
            sentence_seg = ws([t])[0]
        elif WORD_SEGMENTER == "jieba":
            sentence_seg = jieba.lcut(t)
        elif WORD_SEGMENTER == "mix_0":
            temp = ws([t])[0]
            sentence_seg = []
            for seg_t in temp:
                sentence_seg += jieba.lcut(seg_t)
        elif WORD_SEGMENTER == "mix_1":
            temp = jieba.lcut(t)
            sentence_seg = []
            for seg_t in temp:
                sentence_seg += ws([seg_t])[0]
    
        sentence_seg = [seg_t for seg_t in sentence_seg if seg_t!=' ']
        seg_texts = ' '.join(sentence_seg)
        
        data["corpus"] += [seg_texts]
        data["label"] += [label]
        data["src_texts"] += [t]
        data["src_label"] += [src_labels[label]]
    return data, num_labels # Dict[List], List[Int]

# data preprocess
data_train, num_labels = get_model_data(f"data/{DATASET}/train.tsv")
data_test, _ = get_model_data(f"data/{DATASET}/test.tsv")

# model
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
train_tfidf = tfidf_vectorizer.fit_transform(data_train["corpus"])
idf = tfidf_vectorizer.idf_
voc_list = tfidf_vectorizer.get_feature_names()

test_tfidf = get_tfidf(data_test["corpus"], voc_list, idf)

# SIM_MODE = 0
sim_array = get_similarity(test_tfidf, train_tfidf)
acc = get_prediction(sim_array, data_test, data_train, 0)
print(f"{DATASET} {WORD_SEGMENTER}")
print(f"SIM_MODE_0\t{acc:.6f}")

# SIM_MODE = 1
sim_array = get_similarity(test_tfidf, train_tfidf, num_labels)
acc = get_prediction(sim_array, data_test, data_train, 1)
print(f"SIM_MODE_1\t{acc:.6f}")

