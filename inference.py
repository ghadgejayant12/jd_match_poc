import os
from gensim.models import Doc2Vec, Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import string
import PyPDF2
from scipy.spatial.distance import cosine,euclidean
import numpy as np
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
# Here I have Implemented 3 methods :
# 1. Skills extraction and TF-IDF based filtering
# 2. Trained a Word2Vec model using a corpus of 30,000 resumes, then created a document vector by calculating mean
# 	 of the word vectors.
# 3. TF-IDF based document vector calculation

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
nlp2 = spacy.load('en_core_web_sm')
ruler = nlp2.add_pipe("entity_ruler")
ruler.from_disk('jz_skill_patterns.jsonl')
print('now loading the Word2Vec model')
word_model = Word2Vec.load('basic_resume_word2vec.model')
print('done loading')
#doc_model = Doc2Vec.load('basic_resume_doc2vec.model')
stop_words = set(stopwords.words('english')+list(string.punctuation))


def preprocess(document):
	tokens=word_tokenize(document)
	filtered_toks = [token.lower() for token in tokens if token.lower() not in stop_words]
	document = ' '.join(filtered_toks)
	document = nlp(document)
	result=list()
	for tok in document:
		result.append(tok.lemma_)
	return result


def get_dvec(model,text):
	words = [model.wv[word] for word in text if word in list(model.wv.key_to_index.keys())]
	sentence_vector = np.zeros(model.vector_size)
	if len(words)>0:
		sentence_vector = np.mean(words, axis=0)
	return sentence_vector

# import torch
# from transformers import pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# summarizer = pipeline("summarization")
# sent_encoder = pipeline("feature-extraction", model="sentence-transformers/bert-base-nli-mean-tokens")
#
# import spacy
# import re
# def preprocess(document_text, language='en'):
#     document_text = re.sub(r'http\S+', '', document_text)
#     document_text = re.sub(r'\d+',' ',document_text)
#     document_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','',document_text)
#     nlp = spacy.load(f'{language}_core_web_sm')
#     doc = nlp(document_text)
#     cleaned_doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
#     lemmatized_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#     print(lemmatized_doc)
#     return ' '.join(lemmatized_doc)
#
# def compute_similarity(doc1, doc2):
#     doc1 = preprocess(doc1)
#     doc2 = preprocess(doc2)
#     summary1 = summarizer(doc1, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
#     summary2 = summarizer(doc2, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
#     embeddings1 = sent_encoder(summary1, padding=True, truncation=False)
#     embeddings2 = sent_encoder(summary2, padding=True, truncation=False)
#     embeddings1 = embeddings1[0]
#     embeddings2 = embeddings2[0]
#     embeddings1 = torch.Tensor(embeddings1).numpy()
#     embeddings2 = torch.Tensor(embeddings2).numpy()
#     similarity_score = cosine_similarity(embeddings1, embeddings2).mean()
#     return similarity_score


class MakeInference:
	def __init__(self, resume_path, results=10):
		self.resume_path = resume_path
		self.results = results
		self.doc_vecs = dict() # this will contain key:value --> document_path:document_vector
		self.extractor=None # this will store the TF-IDF model for skills

	def load_docs(self):
		docs = os.listdir(self.resume_path)
		for d in docs:
			print('Currently Handling :',d)
			path=self.resume_path+'/'+d
			resume=open(path,'rb')
			print('Resume opened')
			reader=PyPDF2.PdfReader(resume)
			print('PDF loaded into program')
			text=''
			for page in reader.pages:
				pageobj = page
				text=text+' '+pageobj.extract_text()
			print('All the pages loaded into a string')
			print(type(text))
			text = preprocess(text)
			print('Resume data cleaned')
			vector = get_dvec(word_model, text)
			print('Vector for this document created')
			self.doc_vecs[path]=vector
			resume.close()
			print('Closing the file -----------------------------------------')
		return True

	def rank_jd(self, jd_text):
		jd_text = preprocess(jd_text)
		jd_vec = get_dvec(word_model, jd_text)
		result = list()
		for key in self.doc_vecs.keys():
			print(key)
			res_vec = self.doc_vecs[key]
			dist = euclidean(jd_vec,res_vec)
			result.append((dist,key))
		result.sort()
		return result

	def load_docs_skill(self):
		docs = os.listdir(self.resume_path)
		documents=dict()
		result=dict()
		for d in docs:
			path=self.resume_path+'/'+d
			resume=open(path,'rb')
			reader=PyPDF2.PdfReader(resume)
			text=''
			for page in reader.pages:
				text=text+' '+page.extract_text()
			text=' '.join(preprocess(text))
			skills=list()
			d1=nlp2(text)
			for ent in d1.ents:
				if ent.label_=='SKILL':
					skills.append(ent.text)
			skills = list(set(skills))
			skills = ', '.join(skills)
			documents[path] = skills
		extractor = TfidfVectorizer()
		extractor.fit(list(documents.values()))
		for key in documents.keys():
			txt = documents[key]
			vec = extractor.transform([txt])
			self.doc_vecs[key] = {'vector':vec, 'skills':txt}
		self.extractor = extractor
		return True

	def rank_jd_skills(self,jd_text):
		jd_text = preprocess(jd_text)
		jd_text = ' '.join(jd_text)
		doc1 = nlp2(jd_text)
		skills = list()
		for ent in doc1.ents:
			if ent.label_== 'SKILL':
				skills.append(ent.text)
		skills = ' '.join(skills)
		vec_jd = self.extractor.transform([skills])
		results = list()
		for key in self.doc_vecs.keys():
			#print(key)
			key1 = key.split('/')[-1]
			print(key1)
			vec1 = self.doc_vecs[key]['vector']
			skill = self.doc_vecs[key]['skills']
			print(vec_jd.toarray().shape)
			dist = cosine(vec_jd.toarray()[0], vec1.toarray()[0])
			#results.append((1-dist, key, skills))
			#skill=','.join(list(set(skill)))
			results.append([int((1-dist)*100), key, skill, key1])
		return results