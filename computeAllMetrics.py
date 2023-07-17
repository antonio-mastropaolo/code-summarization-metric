import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
from nltk.translate.bleu_score import sentence_bleu
import nltk
import sacrebleu
nltk.download('punkt')

def cosine_similarity_score(x, y):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity_matrix = cosine_similarity(x, y)
    return cosine_similarity_matrix

def euclidean_distance_score(x, y):
    from sklearn.metrics.pairwise import euclidean_distances
    euclidean_distance_matrix = euclidean_distances(x, y)
    return euclidean_distance_matrix

def f1_score(p, r):
    return ((2*p*r)/(p+r))


def chrF_score(dataframe, average=None):
    
    chrf_scores = []
    for idx,row in dataframe.iterrows():
        hypothesis = ' '.join(row['codeComment'].split(' '))
        reference = ' '.join(row['originalComment'].split(' '))
        chrf_score = sacrebleu.sentence_chrf(hypothesis, [reference])
        chrf_scores.append(chrf_score.score/100)
    
    dataframe['chrf_score'] = chrf_scores
    return dataframe



def jaccard_similarity_score(dataframe, average=None):
    score_dict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        refwords = set(ref.strip().split())
        predwords = set(pred.strip().split())
        intersection = refwords.intersection(predwords)
        union = refwords.union(predwords)
        score_dict.append(float(len(intersection)/len(union)))
    
    dataframe['jaccard'] = score_dict
    return dataframe


def indv_bleu_score(dataframe):

    badict = []
    b1dict = []
    b2dict = []
    b3dict = []
    b4dict = []

    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        badict.append(sentence_bleu([ref], pred))
        b1dict.append(sentence_bleu([ref], pred, weights=(1,0,0,0)))
        b2dict.append(sentence_bleu([ref], pred, weights=(0,1,0,0)))
        b3dict.append(sentence_bleu([ref], pred, weights=(0,0,1,0)))
        b4dict.append(sentence_bleu([ref], pred, weights=(0,0,0,1)))
    
    dataframe['bleu-A'] = badict
    dataframe['bleu-1'] = b1dict
    dataframe['bleu-2'] = b2dict
    dataframe['bleu-3'] = b3dict
    dataframe['bleu-4'] = b4dict
    return dataframe


def indv_rouge_score(dataframe):
    import rouge
    # rougedict1 = []
    # rougedict2 = []
    # rougedict3 = []
    # rougedict4 = []
    # rougedictL = []
    # rougedictW = []

    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True,
                            length_limit=100, length_limit_type='words', alpha=.5, weight_factor=1.2)
        score = evaluator.get_scores(ref, pred)
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-1-f1'] = score['rouge-1']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-1-p'] = score['rouge-1']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-1-r'] = score['rouge-1']['r']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-2-f1'] = score['rouge-2']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-2-p'] = score['rouge-2']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-2-r'] = score['rouge-2']['r']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-3-f1'] = score['rouge-3']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-3-p'] = score['rouge-3']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-3-r'] = score['rouge-3']['r']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-4-f1'] = score['rouge-4']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-4-p'] = score['rouge-4']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-4-r'] = score['rouge-4']['r']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-l-f1'] = score['rouge-l']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-l-p'] = score['rouge-l']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-l-r'] = score['rouge-l']['r']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-w-f1'] = score['rouge-w']['f']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-w-p'] = score['rouge-w']['p']
        dataframe.loc[dataframe['codeComment'] == pred, 'rouge-w-r'] = score['rouge-w']['r']

        # print(evaluator.get_scores(ref, pred))
        # sys.exit(-1)
        # rougedict.append(score)

    #dataframe['rouge-l-f1'] = rougedict
    return dataframe


def indv_meteor_score(dataframe):
    from nltk.translate.meteor_score import single_meteor_score
    msdict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            msdict.append(0)
            continue
        ms = single_meteor_score(ref, pred)
        msdict.append(ms)

    dataframe['meteor'] = msdict
    return dataframe


def tfidf_vectorizer(dataframe):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cosine_score_dict = []
    euclidean_distance_dict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            cosine_score_dict.append(0)
            continue
        data = [ref, pred]
        vect = TfidfVectorizer()
        vector_matrix = vect.fit_transform(data)
        #print(vector_matrix[0].todense())
        #print("*********")
        #print(vector_matrix[0])
        #sys.exit(-1)
        css = cosine_similarity_score(np.asarray(vector_matrix[0].todense()), np.asarray(vector_matrix[1].todense()))[0][0]
        cosine_score_dict.append(css)
        
        ess = euclidean_distance_score(np.asarray(vector_matrix[0].todense()), np.asarray(vector_matrix[1].todense()))[0][0]
        euclidean_distance_dict.append(ess)
   
    dataframe['tfidf_cosine'] = cosine_score_dict
    dataframe['tfidf_euclidean'] = euclidean_distance_dict

    return dataframe


def indv_universal_sentence_encoder_dict(dataframe):
    import tensorflow_hub as tfhub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = tfhub.load(module_url)
    cosine_score_dict = []
    euclidean_distance_dict = []

    for ref, pred in tqdm(zip(dataframe['originalComment'], dataframe['codeComment'])):
        ref = ref.strip()
        pred = pred.strip()
        print(ref, pred)
        if pred == '':
            cosine_score_dict.append(0)
            continue    
        data = [ref, pred]
        data_emb = model(data)
        data_emb = np.array(data_emb)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        cosine_score_dict.append(css)
        
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        euclidean_distance_dict.append(ess)

    dataframe['USE_cosine_similarity'] = cosine_score_dict
    dataframe['USE_euclidean_distance'] = euclidean_distance_dict
    return dataframe

def official_bert_score(dataframe):
    from bert_score import score

    p_bert = []
    r_bert = []
    f1_bert = []
   
    p, r, f1 = score(list(dataframe['codeComment']), list(dataframe['originalComment']), lang='en', rescale_with_baseline=True)
    for pscore, rscore, f1score in zip(p.numpy(), r.numpy(), f1.numpy()):
        p_bert.append(pscore)
        r_bert.append(rscore)
        f1_bert.append(f1score)


    dataframe['bert-score-precision'] = p_bert
    dataframe['bert-score-recall'] = r_bert
    dataframe['bert-score-f1'] = f1_bert
    return dataframe

def sentence_bert_encoding(dataframe):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('stsb-roberta-large')
    cosine_score_dict = []
    euclidean_distance_dict = []

    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            cosine_score_dict.append(0)
            continue
        data = [ref, pred]
        data_emb = model.encode(data)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        
        cosine_score_dict.append(css)
        euclidean_distance_dict.append(ess)
    
    dataframe['sentence_bert_cosine_similarity'] = cosine_score_dict
    dataframe['sentence_bert_euclidean_distance'] = euclidean_distance_dict
    return dataframe



def infersent_encoding(dataframe):
    import torch
    from infersent.models import InferSent
    model_version = 1
    model_path = "infersent/encoder/infersent%s.pkl" % model_version
    params_model = {'bsize':64, 'word_emb_dim':300, 'enc_lstm_dim':2048, 'pool_type':'max',
                    'dpout_model':0.0, 'version':model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(model_path))
    glove_path = 'infersent/GloVe/glove.840B.300d.txt'
    fasttext_path = 'infersent/fastText/crawl-300d-2M.vec'
    w2v_path = glove_path if model_version == 1 else fasttest_path
    model.set_w2v_path(w2v_path)
    model.build_vocab_k_words(K=100000)
    cosine_score_dict = []
    euclidean_distance_dict = []

    for ref, pred in tqdm(zip(dataframe['originalComment'], dataframe['codeComment'])):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            cosine_score_dict.append(0)
            continue    
        data = [ref, pred]
        data_emb = model.encode(data)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        
        cosine_score_dict.append(css)
        euclidean_distance_dict.append(ess)
    
    dataframe['infersent_cosine_similarity'] = cosine_score_dict
    dataframe['infersent_euclidean_distance'] = euclidean_distance_dict
    return dataframe

#the model is not available
# def attendgru_embedding(dataframe):
#     import tensorflow as tf
#     from tensorflow import keras
#     import tokenizer

#     flatgru_css_dict = dict()
#     flatgru_ess_dict = dict()

#     refs = list()
#     preds = list()
#     for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
#         refs.append('<s> ' + ref.strip() + ' </s>')
#         preds.append('<s> ' + pred.strip() + ' </s>')

#     comstok = pickle.load(open('coms.tok', 'rb'), encoding='UTF-8')
#     fmodelfname = 'attendgru_E01_1612205848.h5'
#     fmodel = keras.models.load_model(fmodelfname, custom_objects={"tf":tf, "keras":keras})
#     dat_input = fmodel.get_layer('input_1')
#     com_input = fmodel.get_layer('input_2')
#     tdats_emb = fmodel.get_layer('embedding')
#     tdats_gru = fmodel.get_layer('gru')
#     dec_emb = fmodel.get_layer('embedding_1')
#     dec_gru = fmodel.get_layer('gru_1')
#     attn_dot = fmodel.get_layer('dot')
#     attn_actv = fmodel.get_layer('activation')
#     attn_context = fmodel.get_layer('dot_1')

#     reftok = comstok.texts_to_sequences(refs)[:, :13]
#     predtok = comstok.texts_to_sequences(preds)[:, :13]
#     ref_input = com_input(reftok)
#     pred_input = com_input(predtok)
#     ref_emb = dec_emb(ref_input)
#     ref_gru = dec_gru(ref_emb)
#     pred_emb = dec_emb(pred_input)
#     pred_gru = dec_gru(pred_emb)
#     flatp = tf.keras.layers.Flatten()(pred_gru)
#     flatr = tf.keras.layers.Flatten()(ref_gru)
    
#     for ref, pred in zip(flatr, flatp):
#         css = cosine_similarity_score([ref], [pred])[0][0]
#         ess = euclidean_distance_score([ref], [pred])[0][0]
#         flatgru_css_dict.append(css)
#         flatgru_ess_dict.append(ess)

#     dataframe['attendgru_cosine_similarity'] = flatgru_css_dict
#     dataframe['attendgru_euclidean_distance'] = flatgru_ess_dict
#     return dataframe


def main():
    df_data = pd.read_csv('human-annotated-dataset-with-metrics.csv')
    
    print("******************* Computing JACCARD SCORES *******************\n")
    df_data = jaccard_similarity_score(df_data)

    print("******************* Computing BLEU SCORES *******************\n")
    df_data = indv_bleu_score(df_data)
    
    print("******************* Computing ROUGE SCORES *******************\n")
    df_data = indv_rouge_score(df_data)

    print("******************* Computing METEOR SCORES *******************\n")
    df_data = indv_meteor_score(df_data)
    
    print("******************* Computing TF-IDF SCORES *******************\n")
    df_data = tfidf_vectorizer(df_data)
    
    print("******************* Computing USE SCORES *******************\n")
    df_data = indv_universal_sentence_encoder_dict(df_data)

    print("******************* Computing chrF SCORES *******************\n")
    df_data = chrF_score(df_data)
    
    """
    print("******************* Computing BERT SCORES *******************\n")
    df_data = official_bert_score(df_data)

    print("******************* Computing Sentence BERT ENCODING SCORES *******************\n")
    df_data = sentence_bert_encoding(df_data)

    print("******************* Computing InferSent SCORES *******************\n")
    df_data = infersent_encoding(df_data)
    
    # The pre-trained model is not available for download
    #print("******************* Computing AttendGRU SCORES *******************\n")
    #df_data = attendgru_embedding(df_data)
    """
    df_data.to_csv('human-annotated-dataset-with-metrics.csv')


if __name__ == '__main__':
    main()

