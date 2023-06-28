# Approach implemented here :
# 1. Summarization using BERT base then embeddings are generated on the summary using Bart.
# 2.
# 3.
import torch
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import os
import PyPDF2
import pinecone
import numpy as np
summarizer = pipeline("summarization")
sent_encoder = pipeline("feature-extraction", model="sentence-transformers/bert-base-nli-mean-tokens")
pinecone.init(api_key='dffb67e6-d9d4-48c3-a512-e925d41a21d4', environment='us-west1-gcp')
index = pinecone.Index('resume-mapping')
nlp2 = spacy.load('en_core_web_sm')
ruler = nlp2.add_pipe('entity_ruler')
ruler.from_disk('jz_skill_patterns.jsonl')


def preprocess(document_text, language='en'):
    document_text = re.sub(r'http\S+', '', document_text)
    document_text = re.sub(r'\d+',' ',document_text)
    document_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','',document_text)
    nlp = spacy.load(f'{language}_core_web_sm')
    doc = nlp(document_text)
    cleaned_doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
    lemmatized_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    print(lemmatized_doc)
    return ' '.join(lemmatized_doc)


def compute_similarity(doc1, doc2):
    doc1 = preprocess(doc1)
    doc2 = preprocess(doc2)
    summary1 = summarizer(doc1, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    summary2 = summarizer(doc2, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    embeddings1 = sent_encoder(summary1, padding=True, truncation=False)
    embeddings2 = sent_encoder(summary2, padding=True, truncation=False)
    embeddings1 = embeddings1[0]
    embeddings2 = embeddings2[0]
    embeddings1 = torch.Tensor(embeddings1).numpy()
    embeddings2 = torch.Tensor(embeddings2).numpy()
    similarity_score = cosine_similarity(embeddings1, embeddings2).mean()
    return similarity_score


def compute_vector(document):
    document = preprocess(document)
    summary = document
    if len(document)>=100:
        summary = summarizer(document, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    embeddings = sent_encoder(summary, padding=True, truncation=False)
    embeddings = embeddings[0]
    embeddings = torch.Tensor(embeddings).numpy()
    embeddings = np.mean(embeddings, axis=0)
    return embeddings, embeddings.shape


class MakeInference:
    def __init__(self, resume_path):
        self.resume_path = resume_path
        self.skills = dict()

    def load_docs(self):
        files = os.listdir(self.resume_path)
        for file in files:
            path = self.resume_path+'\\'+file
            resume = open(path, 'rb')
            reader = PyPDF2.PdfReader(resume)
            text = ''
            for page in reader.pages:
                text = text+' '+page.extract_text()
            vector, shape = compute_vector(text)
            skills = list()
            d1 = nlp2(text)
            for ent in d1.ents:
                if ent.label_ == 'SKILL':
                    skills.append(ent.text)
            skills = '|'.join(list(set(skills)))
            index.upsert([
                (path, [float(i) for i in list(vector)], {'shape': str(vector.shape), 'skills': skills})
            ])
        return True

    def rank_jd(self, jd_text):
        jd_text = preprocess(jd_text)
        vector_jd, shape = compute_vector(jd_text)
        results = list()
        vector_jd = [float(i) for i in list(vector_jd)]
        matches = index.query(
            vector=vector_jd,
            top_k=10,
            include_values=True,
            include_metadata=True
        )
        for match in matches['matches']:
            print(match)
            key = match['id']
            skill = match['metadata']['skills']
            score = match['score']
            results.append([int(score*100), key, skill, key])
        return results
