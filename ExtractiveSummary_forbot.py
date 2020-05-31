"""
This project aims to perform extractive summaries on articles
"""

import os
os.chdir('/Users/jaoming/Documents/Active Projects')
# Extracting pre-trained word-vectors
word_embeddings = {} # these are vectors that represent words
g_file = open('glove.6B.100d.txt', encoding = 'utf-8')
for line in g_file:
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype = 'float32')
       word_embeddings[word] = coefs
g_file.close()

os.chdir('/Users/jaoming/Documents/Active Projects/Text Summarisation')

import requests                                         # to get html source code
from bs4 import BeautifulSoup as bsoup                  # to parse the html source code
from bs4.element import Tag                             # to separate link and text

import numpy as np                                      # array manipulation
import spacy                                            # nlp module
from spacy.lang.en.stop_words import STOP_WORDS
import re                                               # text cleaning
import networkx as nx                                   # matrix manipulation
from sklearn.metrics.pairwise import cosine_similarity  # similarity computation

# Custom NLP parser (not used for this case)
def set_custom_boundaries(doc):
       # adds support to parse quotes as sentences
       for token in doc[:-1]:
              if token.text == "-":
                     doc[token.i+1].is_sent_start = False
       return doc

custom_nlp = spacy.load('en_core_web_sm')
custom_nlp.add_pipe(set_custom_boundaries, before = 'parser')

# Extract the HTML source code from the link
def extract_html_code(link):
       html_code = requests.get(link)
       html_code = html_code.text
       return html_code

# Extract the story from the source code
def extract_html_text(html_code):
       soup = bsoup(html_code, features = 'html.parser')
       text = [i.contents for i in soup.findAll('p') if len(i.attrs) == 0][:-1]
       return text

# Collating any links in the article and cleaning up the text
def clean_html_text(text):
       article_links = []
       clean_text = ""

       for sentence in text:
              if len(sentence) == 1:
                     temp = sentence[0]
                     if type(temp) != Tag:
                            clean_text += temp + " "
              else:
                     temp = ""
                     for parts in sentence:
                            if type(parts) == Tag:
                                   temp += parts.contents[0] + " "
                                   article_links.append(parts['href'])
                            else:
                                   temp += parts
       return clean_text, article_links

# Main Class
class ExtractSummary():
       """
       This class aims to create extractive summaries of articles
       """
       def extracting_sentences(self, doc):
              """
              Function:     Extracts the sentences from the document and cleans them up before returning them

              Inputs:       Document

              Returns:      Cleaned individual sentences
              """
              dirty_sentences = [i.text.strip() for i in doc.sents]
              clean_sentences = []
              for dirt in dirty_sentences:
                     # un-capitalise the words in the sentence and remove stop words
                     clean = " ".join([i.lower() for i in dirt if i not in STOP_WORDS])
                     
                     # removing anything that's not a word
                     clean = re.sub("[^a-zA-Z0-9-]", " ", clean).strip()
                     clean = re.sub("\s+", " ", clean)
                     # clean = re.sub("\s-\s", "-", clean)
                     clean_sentences.append(clean)
              return clean_sentences, dirty_sentences

       def convert_to_sentence_vectors(self, sentences):
              """
              Function:     Converts sentences into a vector form by summing up the word vectors in that sentence and dividing it over the number of words in that sentence

              Inputs:       Sentences

              Returns:      A list of sentence vectors
              """
              sentence_vectors = []
              for sentence in sentences:
                     sent_len = len(sentence.split())
                     if sent_len != 0:
                            vect = sum([word_embeddings.get(word, np.zeros((100, ))) for word in sentence.split()])
                            vect = vect / (sent_len + 0.001)
                     else:
                            vect = np.zeros((100, ))
                     sentence_vectors.append(vect)
              return sentence_vectors

       def matrix_preparation(self, sent_vects, num_sentences):
              """
              Function:     Prepares the sentence ranking matrix for optimisation

              Inputs:       Sentence vectors

              Returns:      Matrix
              """
              similarity_matrix = np.zeros([num_sentences, num_sentences])

              for i in range(num_sentences):
                     for j in range(num_sentences):
                            if i != j:
                                   similarity_matrix[i][j] = cosine_similarity(sent_vects[i].reshape(1, 100), sent_vects[j].reshape(1, 100))[0, 0]
              return similarity_matrix

       def summarise(self, article):
              """
              Function:     Main function of the class that executes the summarising

              Inputs:       The article in text form

              Returns:      Extracted Summary (Top 5 sentences)
              """
              # Parsing the article into a NLP instance
              doc = custom_nlp(article) # choose which parser you want to use

              # Extracting clean individual sentences
              sentences, original_sentences = self.extracting_sentences(doc)
              n_sentences = len(sentences)

              # Converting sentences into vectors
              sentence_vectors = self.convert_to_sentence_vectors(sentences)

              # Creating the similarity matrix for the sentences
              sim_mat = self.matrix_preparation(sentence_vectors, n_sentences)

              # Applying the page-ranking algorithm to the matrix
              nx_graph = nx.from_numpy_array(sim_mat)
              scores = nx.pagerank(nx_graph)

              # Extracting the sentence and scores
              ranked_sentences = sorted([(scores[i], i, s) for i, s in enumerate(original_sentences)], reverse = True)

              return ranked_sentences

# Main function
def main(link):
       es = ExtractSummary()

       html_code = extract_html_code(link)
       text = extract_html_text(html_code)
       clean_text, article_links = clean_html_text(text)

       ranked_sentences = es.summarise(clean_text)
       summary_sentences = ranked_sentences[:5] # len(ranked_sentences)//3]              # can tweak this for the length of the summary
       summary_sentences.sort(key = lambda x: x[1])
       summary = "<b>Summary of the article:</b>\n"
       for i in summary_sentences:
              summary += i[2] + " "

       full_length = len(clean_text)
       summary_length = len(summary) - 30

       summary += "\n\n<b>Links that were in the article:</b>"
       for links in article_links:
              summary += '\n' + links + '\n'

       summary += "\n<b>Stats:</b>"
       summary += "\nThis bot has churned out a <b>" + str(int(1000*summary_length/full_length)/10) + "%</b> summary of the article"
       
       return summary

