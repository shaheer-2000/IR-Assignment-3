# # WIP! USE tf-idf.py separately

# from fetcher import Fetcher
# from parser import Parser
# from lexical_chains import LexicalChains

# import numpy as np
# from sklearn.model_selection import train_test_split

# fetcher = Fetcher()
# parser = Parser()

# fetcher.cd_dir("course-cotrain-data/fulltext/course")

# X = np.ndarray((2,))
# y = np.array([])

# courses = []
# i = 0
# for doc in fetcher.get_files():
# 	parsed_text = parser.get_text(fetcher.read_file(doc))
# 	courses.append(parsed_text)

# 	lexical_chains = LexicalChains()
# 	lexical_chains.construct_chains(parsed_text)
# 	np.append(X, lexical_chains.get_features(), axis=0)
# 	np.append(y, [1], axis=0)
# 	if i > 5:
# 		break
# 	i += 1

# fetcher.cd_dir("course-cotrain-data/fulltext/non-course")

# non_courses = []
# i = 0
# for doc in fetcher.get_files():
# 	parsed_text = parser.get_text(fetcher.read_file(doc))
# 	non_courses.append(parsed_text)

# 	lexical_chains = LexicalChains()
# 	lexical_chains.construct_chains(parsed_text)
# 	np.append(X, lexical_chains.get_features())
# 	np.append(y, [0])
# 	if i > 5:
# 		break
# 	i += 1

# print(X.shape, y.shape, X[:5])


import pickle
from pathlib import Path
import copy as cp

def save(file, obj):
	_path = Path(file)
	_dir = _path.parents[0]

	if not _dir.exists() or not _dir.is_dir():
		_dir.mkdir(parents=True)

	with _path.open("wb") as f:
		pickle.dump(obj, f)

def load(file):
	_path = Path(file)
	_dir = _path.parents[0]

	if not _dir.exists() or not _dir.is_dir():
		raise FileNotFoundError("invalid path for loading pickle object")

	with _path.open("rb") as f:
		return pickle.load(f)

from fetcher import Fetcher
from parser import Parser
from preprocessor import PreProcessor
from index_builder import IndexBuilder
from vectorizer import Vectorizer

fetcher = Fetcher()
parser = Parser()
preprocessor = PreProcessor()

try:
	X, y = load("./dumps/training_data.pickle")
except FileNotFoundError:
	X, y = [], []

	fetcher.cd_dir("course-cotrain-data/fulltext/course")
	for doc in fetcher.get_files():
		parsed_text = parser.get_text(fetcher.read_file(doc))
		X.append(preprocessor.get_tokens(parsed_text))
		y.append(1)

	fetcher.cd_dir("course-cotrain-data/fulltext/non-course")
	for doc in fetcher.get_files():
		parsed_text = parser.get_text(fetcher.read_file(doc))
		X.append(preprocessor.get_tokens(parsed_text))
		y.append(0)

	save("./dumps/training_data.pickle", (X, y))

try:
	index_builder = load("./dumps/index.pickle")
except FileNotFoundError:
	index_builder = IndexBuilder()
	index_builder.build_index(X)
	save("./dumps/index.pickle", index_builder)

if __name__ == "__main__":
	# temp
	# import numpy as np
	# tfidf_vector = index_builder.get_vector()
	# print(tfidf_vector[np.nonzero(tfidf_vector)])

	# constants
	TOP_TFIDF_TERMS_THRESHOLD = 100
	TFIDF = "tfidf"
	IS_NOUN = "is_noun"

	# for sorting by tfidf, and filtering nouns
	doc_features = {}
	
	for term in index_builder.term_dict:
		postings_list = index_builder.term_dict[term].get_postings_list()
		for doc_index in postings_list:
			if doc_index not in doc_features:
				doc_features[doc_index] = {}

			doc_features[doc_index][term] = {
				TFIDF: index_builder.term_dict[term].get_tfidf(doc_index),
				IS_NOUN: index_builder.term_dict[term].is_noun
			}

	# doc_features have been built, no need for index now
	# discarding index, to free up memory, can be loaded later anyways
	del index_builder # wait for gc, rather than explicit gc.collect()

	# sort by tfidf
	for doc_index in doc_features:
		doc_features[doc_index] = dict(sorted(doc_features[doc_index].items(), key=lambda item: item[1][TFIDF], reverse=True))

	# keep only top 100 terms by tfidf for each doc
	for doc_index in doc_features:
		i = 0
		features = cp.deepcopy(doc_features[doc_index])
		for feature in features:
			if i < TOP_TFIDF_TERMS_THRESHOLD:
				i += 1
				continue

			doc_features[doc_index].pop(feature, None)
			i += 1

	# now we know all of the features we want
	# we can sort them alphabetically and make a tf-idf vector now
	# we want to keep the feature names for this to build more vectors later
	tfidf_vectorizer = Vectorizer()
	try:
		X = load("./dumps/tfidf_1_training_vector")
	except FileNotFoundError:
		X = tfidf_vectorizer.fit_transform(doc_features)
		save("./dumps/tfidf_1_training_vector", X)

	# temp
	from sklearn.model_selection import train_test_split
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.metrics import confusion_matrix, classification_report

	X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0)
	model = MultinomialNB().fit(X_train, y_train)
	# score = model.score(X_dev, y_dev)

	y_pred = model.predict(X_dev)

	print("Confusion Matrix")
	print(confusion_matrix(y_true=y_dev, y_pred=y_pred))

	print("Classification Report")
	print(classification_report(y_true=y_dev, y_pred=y_pred, zero_division=0))

