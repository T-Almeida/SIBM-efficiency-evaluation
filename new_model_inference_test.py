import time
import argparse
import os

from collections import defaultdict

from mmnrm.dataset import TestCollectionV2
from mmnrm.utils import set_random_seed, load_neural_model, save_model_weights, load_model, load_sentence_generator, flat_list

import numpy as np
import nltk

from timeit import default_timer as timer


import tensorflow as tf

# fix the generator randoness
set_random_seed()

def build_data_generators(tokenizer, queries_sw=None, docs_sw=None):
    
    def maybe_tokenize(documents):
        if "tokens" not in documents:
            split = nltk.sent_tokenize(documents["text"])
            documents["tokens"] = tokenizer.texts_to_sequences(split)
            if docs_sw is not None:
                for tokenized_sentence in documents["tokens"]:
                    tokenized_sentence = [token for token in tokenized_sentence if token not in docs_sw]
    
    def train_generator(data_generator):
        while True:

            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)

            # tokenization, this can be cached for efficientcy porpuses NOTE!!
            tokenized_query = tokenizer.texts_to_sequences(query)

            if queries_sw is not None:
                for tokens in tokenized_query:
                    tokenized_query = [token for token in tokens if token not in queries_sw] 
            
            saveReturn = True
            
            for batch_index in range(len(pos_docs)):
                
                # tokenizer with cache in [batch_index][tokens]
                maybe_tokenize(pos_docs[batch_index])
                
                # assertion
                if all([ len(sentence)==0  for sentence in pos_docs[batch_index]["tokens"]]):
                    saveReturn = False
                    break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                          # for obvious reasons
                
                maybe_tokenize(neg_docs[batch_index])
                
            if saveReturn: # this is not true, if the batch is rejected
                yield tokenized_query, pos_docs, neg_docs

    def test_generator(data_generator):
        for _id, query, docs in data_generator:
            tokenized_queries = []
            for i in range(len(_id)):
                # tokenization
                tokenized_query = tokenizer.texts_to_sequences([query[i]])[0]

                if queries_sw is not None:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
                
                tokenized_queries.append(tokenized_query)
                    
        
                for doc in docs[i]:
                    maybe_tokenize(doc)
                                                 
            yield _id, tokenized_queries, docs
            
    return train_generator, test_generator

def model_train_generator_for_model(model):

    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    train_gen, test_gen = build_data_generators(model.tokenizer)
    
    pad_tokens = lambda x, max_len, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_len,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_sentences = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))
    
    def maybe_padding(document):
        if isinstance(document["tokens"], list):
            document["tokens"] = pad_tokens(pad_sentences(document["tokens"], cfg["max_passages"]), cfg["max_p_terms"])
            
    def train_generator(data_generator):
 
        for query, pos_docs, neg_docs in train_gen(data_generator):
            
            query = pad_tokens(query, cfg["max_q_terms"])
            
            pos_docs_array = []
            neg_docs_array = []
            
            # pad docs, use cache here
            for batch_index in range(len(pos_docs)):
                maybe_padding(pos_docs[batch_index])
                pos_docs_array.append(pos_docs[batch_index]["tokens"])
                maybe_padding(neg_docs[batch_index])
                neg_docs_array.append(neg_docs[batch_index]["tokens"])
            
            yield [query, np.array(pos_docs_array)], [query, np.array(neg_docs_array)]
            
    def test_generator(data_generator):
        
        for ids, query, docs in test_gen(data_generator):
            
            docs_ids = []
            docs_array = []
            query_array = []
            query_ids = []
            
            for i in range(len(ids)):
                
                for doc in docs[i]:
                    # pad docs, use cache here
                    maybe_padding(doc)
                    docs_array.append(doc["tokens"])
                    docs_ids.append(doc["id"])
                
                query_tokens = pad_tokens([query[i]], cfg["max_q_terms"])[0]
                query_tokens = [query_tokens] * len(docs[i])
                query_array.append(query_tokens)
                    
                query_ids.append([ids[i]]*len(docs[i]))
            
            yield flat_list(query_ids), [np.array(flat_list(query_array)), np.array(docs_array)], docs_ids, None
            
    return train_generator, test_generator



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('batch_size', type=int, help="size of the batch that will be used during the tests")
    parser.add_argument('-o', type=str, default=None, help="output file to append the results")

    args = parser.parse_args()
    
    # use bioASQ download folder 
    
    os.chdir("../EACL2021-reproducibility")
    ranking_model = load_model("../EACL2021-reproducibility/download_folder/models/olive-haze-9_val_collection0_map@10")
    os.chdir("../PerformanceComparasion")
    _, test_input_generator = model_train_generator_for_model(ranking_model)
    
    def convert_to_tensor(data_generator):
        for query_id, Y, docs_info, offsets_docs in test_input_generator(data_generator):
            yield query_id, [tf.convert_to_tensor(Y[0], dtype=tf.int32), tf.convert_to_tensor(Y[1], dtype=tf.int32)], docs_info, offsets_docs

    
    test_collection = TestCollectionV2\
                                    .load("../EACL2021-reproducibility/download_folder/pickle-data/query_docs_pairs")\
                                    .batch_size(args.batch_size)\
                                    .set_transform_inputs_fn(convert_to_tensor)

    q_scores = defaultdict(list)

    times = []

    @tf.function
    def untraced_static_model(x):
        return ranking_model(x)

    query_id, Y, docs_info, offsets_docs = next(test_collection.generator())
    static_model = untraced_static_model.get_concrete_function(Y)    
    static_model(*Y)

    for i, _out in enumerate(test_collection.generator()):
        query_id, Y, docs_info, offsets_docs = _out
        s_time = timer()

        #scores, q_sentence_attention = rank_model.predict(Y)
        output = static_model(*Y)
        dummy = output[0].numpy()
        times.append(timer()-s_time)

        if not i%10:
            print("Evaluation {} | time {:.3f} | avg {:.3f} +/- {:.3f} | median {:.3f}".format(i, 
                                                                                           times[-1], 
                                                                                           np.mean(times),
                                                                                           np.std(times),
                                                                                           np.median(times)),end="\r")

        #scores = scores[:,0].tolist()
        #q_scores[query_id].extend(list(zip(docs_ids,scores)))
        #for i in range(len(docs_info)):
        #    q_scores[query_id[i]].append((docs_info[i], scores[i], q_sentence_attention[i], offsets_docs[i]))

    # sort the rankings
    #for query_id in q_scores.keys():
    #    q_scores[query_id].sort(key=lambda x:-x[1])
    #    q_scores[query_id] = q_scores[query_id][:10]


    #times = times[2:len(times)-3]

    if args.o is not None:
        with open(args.o, "a") as f:
            f.write("{},{},{:.3f},{:.3f},{:.3f}\n".format("new",
                                                     args.batch_size,
                                                     np.mean(times),
                                                     np.std(times),
                                                     np.median(times)))