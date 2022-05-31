import numpy as np
# import copy as cp
import nltk

TF = "tf"
TFIDF = "tfidf"
IGNORED = "ignored"

class TermNode:
	def __init__(self, is_noun=False):
		self.df = 0
		self.postings_list = {}
		self.is_noun = is_noun
		self.ignored = False

	def add_posting(self, doc):
		if doc not in self.postings_list:
			self.postings_list[doc] = {
				TF: 0,
				TFIDF: 0,
				IGNORED: False
			}
			self.df += 1

		self.postings_list[doc][TF] += 1

	def ignore_doc(self, doc):
		if self.is_ignoring(doc):
			return

		self.df -= 1
		self.postings_list[doc][IGNORED] = True

	def is_ignoring(self, doc):
		return self.postings_list[doc][IGNORED]

	def get_postings_list(self):
		_postings_list = {}
		for doc in self.postings_list:
			if not self.is_ignoring(doc):
				_postings_list[doc] = self.postings_list[doc]

		return _postings_list

	def get_tf_in_doc(self, doc):
		if doc not in self.postings_list or self.is_ignoring(doc):
			return 0

		return self.postings_list[doc][TF]

	def get_df(self):
		# df after filtering ignored docs
		# to get actual df, use len(self.postings_list.keys())
		return self.df

	def get_tfidf(self, doc):
		if doc not in self.postings_list or self.is_ignoring(doc):
			return 0

		return self.postings_list[doc][TFIDF]

	def calc_tfidf(self, doc, doc_count):
		tf = np.log10(1 + self.get_tf_in_doc(doc))
		idf = np.log10(doc_count / self.df)

		self.postings_list[doc][TFIDF] = tf * idf


class IndexBuilder:
	def __init__(self):
		self.term_count = 0
		self.term_dict = {}
		self.vector = None

	def build_index(self, docs):
		doc_count = len(docs)

		doc_index = 0
		for doc in docs:
			tokens = nltk.pos_tag(doc)
			for (token, pos_tag) in tokens:
				if token not in self.term_dict:
					self.term_dict[token] = TermNode(is_noun=pos_tag[:2] == "NN")
					self.term_count += 1

				self.term_dict[token].add_posting(doc_index)

			for token in doc:
				self.term_dict[token].calc_tfidf(doc_index, doc_count)

			doc_index += 1

		# alphabetically sort the term dictionary
		self.term_dict = dict(sorted(self.term_dict.items(), key=lambda item: item[0]))

		self.construct_vector(doc_count)

	# builds a tfidf vector for Multinomial Naive Bayes
	def construct_vector(self, doc_count):
		self.vector = np.zeros((doc_count, self.term_count), np.float32)

		term_index = 0
		for term in self.term_dict:
			postings_list = self.term_dict[term].get_postings_list()

			for doc_index in postings_list:
				self.vector[doc_index][term_index] = self.term_dict[term].get_tfidf(doc_index)

			term_index += 1

	def get_vector(self):
		return self.vector

	def purge_terms(self, min_tfidf=0, only_nouns=False):
		for term in self.term_dict:
			noun_filter = (only_nouns and (not self.term_dict[term].is_noun))
			postings_list = self.term_dict[term].get_postings_list()

			for doc_index in postings_list:
				tfidf_score = self.term_dict[term].get_tfidf(doc_index)
				if tfidf_score < min_tfidf or noun_filter:
					self.term_dict[term].ignore_doc(doc_index)

if __name__ == "__main__":
	from fetcher import Fetcher
	from parser import Parser
	from preprocessor import PreProcessor

	fetcher = Fetcher()
	parser = Parser()
	preprocessor = PreProcessor()

	fetcher.cd_dir("course-cotrain-data/fulltext/course")
	
	X = []
	i = 0
	for doc in fetcher.get_files():
		parsed_text = parser.get_text(fetcher.read_file(doc))
		X.append(preprocessor.get_tokens(parsed_text))

		if i > 100:
			break
		i += 1

	index_builder = IndexBuilder()
	index_builder.build_index(X)

	tfidf_vector = index_builder.get_vector()
	print(tfidf_vector[np.nonzero(tfidf_vector)])
