import numpy as np
from preprocessor import PreProcessor

preprocessor = PreProcessor()

class Vectorizer:
	def __init__(self):
		self.train_vector = None
		self.features = []

	def vectorize(self, content, is_binary=False):
		global preprocessor
		feature_counts = {}
		test_vector = np.zeros((1, len(self.features)), np.float32)

		if is_binary:
			for feature in list(set(preprocessor.get_tokens(content))):
				try:
					feature_index = self.features.index(feature)
				except ValueError:
					continue

				test_vector[0][feature_index] = 1
		else:
			# calc freq
			for feature in preprocessor.get_tokens(content):
				if feature not in feature_counts:
					feature_counts[feature] = 0
				
				feature_counts[feature] += 1

			for feature in feature_counts:
				tf = np.log10(1 + feature_counts[feature])
				# no need to multiply with idf, since it'd be 0

				try:
					feature_index = self.features.index(feature)
				except ValueError:
					continue

				test_vector[0][feature_index] = tf

		return test_vector

	def fit_transform(self, doc_features, is_binary=False):
		for doc in doc_features:
			for feature in doc_features[doc]:
				if feature not in self.features:
					self.features.append(feature)

		# sort features alphabetically
		self.features.sort()

		doc_keys = list(doc_features.keys())
		doc_count = len(doc_keys)
		doc_index_map = {}
		for doc in doc_features:
			doc_index_map[doc] = doc_keys.index(doc)

		feature_count = len(self.features)
		self.train_vector = np.zeros((doc_count, feature_count), np.float32)

		feature_index = 0
		for feature in self.features:
			for doc in doc_features:
				_doc_index = doc_index_map[doc]
				if is_binary:
					self.train_vector[_doc_index][feature_index] = feature in doc_features[doc]
					continue

				if feature in doc_features[doc]:
					self.train_vector[_doc_index][feature_index] = doc_features[doc][feature]["tfidf"]

			feature_index += 1

		return self.train_vector

	def get_feature_names(self):
		return self.features

if __name__ == "__main__":
	train = {
		0: {'homework': {'tfidf': 2.3512639316399904, 'is_noun': True}, 'solution': {'tfidf': 2.3512639316399904, 'is_noun': True}, 'hall': {'tfidf': 1.8191861050085112, 'is_noun': True}, 'hours': {'tfidf': 1.8191861050085112, 'is_noun': True}, 'office': {'tfidf': 1.8191861050085112, 'is_noun': True}, 'phone': {'tfidf': 1.8191861050085112, 'is_noun': True}, 'upson': {'tfidf': 1.8191861050085112, 'is_noun': False}, 'budiu': {'tfidf': 1.4416708791357347, 'is_noun': True}, 'lili': {'tfidf': 1.4416708791357347, 'is_noun': False}, 'operating': {'tfidf': 1.4416708791357347, 'is_noun': False}, 'systems': {'tfidf': 1.4416708791357347, 'is_noun': True}, 'thursday': {'tfidf': 1.4416708791357347, 'is_noun': False}, 'wednesday': {'tfidf': 1.4416708791357347, 'is_noun': False}, 'assignment': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'assignments': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'birman': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'course': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'dynamic': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'filesystem': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'friday': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'group': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'home': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'huang': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'kenneth': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'last': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'lecture': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'linking': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'mihai': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'modified': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'news': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'notes': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'nov': {'tfidf': 0.9095930525042556, 'is_noun': 
True}, 'page': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'practicum': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'prelim': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'programming': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'solutions': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'static': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'structure': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'syllabus': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'system': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'tas': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'tue': {'tfidf': 0.9095930525042556, 'is_noun': True}, 'tuesday': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'unix': {'tfidf': 0.9095930525042556, 'is_noun': False}, 'ychuang': {'tfidf': 0.9095930525042556, 'is_noun': True}}
	}

	vectorizer = Vectorizer()
	X_train = vectorizer.fit_transform(train)
	test = "Too many assignments these days! plus the homework! homework!"
	X_test = vectorizer.tfidf_vectorize(test)

	print(X_train)
	print(X_test)
	print(vectorizer.get_feature_names())