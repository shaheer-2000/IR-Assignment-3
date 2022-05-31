import pickle
from pathlib import Path
import copy as cp

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report

from fetcher import Fetcher
from parser import Parser
from preprocessor import PreProcessor
from index_builder import IndexBuilder
from vectorizer import Vectorizer
from lexical_chains import LexicalChains

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

def main(use_tfidf=True, use_ttc=True, use_lexchains=True, debug=True):
	# Feature Selection One
	USE_TFIDF = use_tfidf
	# Feature Selection Two
	USE_TOPIC_TERMS_COOCCUR = use_ttc
	# Feature Selection Three
	USE_LEXICAL_CHAINS = use_lexchains

	if debug:
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

		# constants
		TOP_TFIDF_TERMS_THRESHOLD = 100
		TOPIC_BASE_SET_THRESHOLD = 50
		TFIDF = "tfidf"
		DF = "df"
		IS_NOUN = "is_noun"

		# for sorting by tfidf, and filtering nouns
		doc_features = {}

		for term in index_builder.term_dict:
			postings_list = index_builder.term_dict[term].get_postings_list()
			for doc_index in postings_list:
				if doc_index not in doc_features:
					doc_features[doc_index] = {}

				doc_features[doc_index][term] = {
					DF: index_builder.term_dict[term].get_df(),
					TFIDF: index_builder.term_dict[term].get_tfidf(doc_index),
					IS_NOUN: index_builder.term_dict[term].is_noun
				}

		# doc_features have been built, no need for index now
		# discarding index, to free up memory, can be loaded later anyways
		del index_builder # wait for gc, rather than explicit gc.collect()


		#--------------------------
		# TF-IDF FEATURE SELECTION
		#--------------------------

		def tfidf_selection(doc_features):
			# _doc_features = cp.deepcopy(_doc_features)

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

			return doc_features
			# return _doc_features

		if USE_TFIDF:
			try:
				doc_features = load(f"./dumps/tfidf_doc_features-{int(USE_TFIDF)}")
			except FileNotFoundError:
				doc_features = tfidf_selection(doc_features)
				save(f"./dumps/tfidf_doc_features-{int(USE_TFIDF)}", doc_features)

		#--------------------------
		# TTC FEATURE SELECTION
		#--------------------------

		# for topic terms co-occurence
		# map doc_features back to a features map, select
		# top 50 most frequent nouns out of these and
		# if word from topic base set appears in same doc
		# as another word from the coll, add that word to 
		# feature space
		def ttc_selection(doc_features):
			feature_space = {}

			for doc_index in doc_features:
				for feature in doc_features[doc_index]:
					_feature = doc_features[doc_index][feature]
					if feature in feature_space:
						continue

					if _feature[IS_NOUN]:
						feature_space[feature] = _feature[DF]

			feature_space = dict(sorted(feature_space.items(), key=lambda item: item[1], reverse=True))

			i = 0
			topic_base_set = cp.deepcopy(feature_space)
			for feature in feature_space:
				if i >= TOPIC_BASE_SET_THRESHOLD:
					topic_base_set.pop(feature, None)

				i += 1

			# now we have a topic base set with top 50 most freq nouns
			# now check whether each element of this set appears in same doc as other
			# elements in the overall collection
			# heuristic for topic-terms co-occurence that is O(n^2)
			_doc_features = cp.deepcopy(doc_features)

			for doc_index in _doc_features:
				contains_base_set_feature = False
				for base_set_feature in topic_base_set:		
					# these are the docs we want to keep
					if base_set_feature in doc_features[doc_index]:
						contains_base_set_feature = True
				
				if not contains_base_set_feature:
					y_index = list(doc_features.keys()).index(doc_index)
					doc_features.pop(doc_index, None)
					# remove from y as well to maintain same size
					try:
						y.pop(y_index)
					except IndexError:
						print(doc_index, len(y), len(doc_features.keys()))
			return doc_features

		if USE_TOPIC_TERMS_COOCCUR:
			try:
				doc_features = load(f"./dumps/ttc_doc_features-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}")
			except FileNotFoundError:
				doc_features = ttc_selection(doc_features)
				save(f"./dumps/ttc_doc_features-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}", doc_features)

			print(len(X), len(y))
			try:
				X, y = load(f"./dumps/training_data-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}.pickle")
			except FileNotFoundError:
				save(f"./dumps/training_data-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}.pickle", (X, y))

		#----------------------------------
		# LEXICAL CHAINS FEATURE SELECTION
		#----------------------------------

		"""
			Construct lexical chains for each doc
			Then only keep features generated by the lexical chain

			If lexical chains are used, vector becomes a binary vector
			and we use BernoulliNB
		"""

		def lexchains_selection(doc_features):
			_doc_features = {}
			a = len(doc_features.keys())
			i = 0
			for doc_index in doc_features:
				doc_tokens = list(doc_features[doc_index].keys())
				lexical_chain = LexicalChains()
				lexical_chain.construct_chains(" ".join(doc_tokens))
				doc_feature_space = lexical_chain.get_features()

				# constructing doc_features for binary vectors
				_doc_features[doc_index] = {}
				for feature in doc_feature_space:
					_doc_features[doc_index][feature] = 1

				print(f"{(i + 1)}/{a} docs chained")
				i += 1

			return _doc_features


		if USE_LEXICAL_CHAINS:
			try:
				doc_features = load(f"./dumps/lexchains_doc_features-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}")
			except FileNotFoundError:
				doc_features = lexchains_selection(doc_features)
				save(f"./dumps/lexchains_doc_features-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}", doc_features)

		# now we know all of the features we want
		# we can sort them alphabetically and make a tf-idf vector now
		# we want to keep the feature names for this to build more vectors later
		
		try:
			vectorizer = load(f"./dumps/vectorizer-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}")
			X = load(f"./dumps/training_vector-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}")
		except FileNotFoundError:
			vectorizer = Vectorizer()
			X = vectorizer.fit_transform(doc_features, is_binary=USE_LEXICAL_CHAINS)
			save(f"./dumps/vectorizer-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}", vectorizer)
			save(f"./dumps/training_vector-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}", X)
	else:
		X = load(f"./dumps/training_vector-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}")
		vectorizer = load(f"./dumps/vectorizer-{int(USE_TFIDF)}-{int(USE_TOPIC_TERMS_COOCCUR)}-{int(USE_LEXICAL_CHAINS)}")

	X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0)
	classifier = None

	if USE_LEXICAL_CHAINS:
		classifier = BernoulliNB()
	else:
		classifier = MultinomialNB()

	model = classifier.fit(X_train, y_train)
	# score = model.score(X_dev, y_dev)

	y_pred = model.predict(X_dev)

	print("Confusion Matrix")
	print(confusion_matrix(y_true=y_dev, y_pred=y_pred))

	print("Classification Report")
	print(classification_report(y_true=y_dev, y_pred=y_pred, zero_division=0))

	return (classifier, vectorizer)
