from sentence_transformers import SentenceTransformer, models, InputExample, losses, util, LoggingHandler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import sys,os
import torch.nn.functional as F
import json
from sentence_transformers import evaluation
from tqdm import tqdm
import logging
import pandas as pd

DEVICE='cuda'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def test(test_data):
    
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

   
    checkPointFolder = "models/triplet-loss/..." #specify the path to the best-performing checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
    model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)

    similarities = []

    for idx,item in tqdm(test_data.iterrows()):

        ################## NEGATIVE ##################
        sentences = []
        sentences.append(item['codeFunctions'])
        sentences.append(item['codeComment'])
        
        
        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sim = util.pytorch_cos_sim(
            sentence_embeddings[0], sentence_embeddings[1]).item()

        similarities.append(sim)

    test_data['SIDE'] = similarities
    test_data.to_csv('human-annotated-dataset-all-metrics.csv')

def main():

    df_response = pd.read_csv('human-annotated-dataset-all-metrics.csv')
    test(df_response)


if __name__ == "__main__":
    main()
