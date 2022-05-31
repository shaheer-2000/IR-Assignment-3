import nltk
from nltk.corpus import stopwords

class PreProcessor:
	def __init__(self):
		self.is_noun = lambda pos: pos[:2] == "NN"
		self.stopwords = stopwords.words("english")

	def tokenize(self, content: str):
		return [token.lower() for token in nltk.word_tokenize(content) if token.isalpha() and len(token) > 2]

	def denoise(self, tokens):
		return [token for token in tokens if token not in self.stopwords]

	def extract_nouns(self, tokens):
		return [noun for (noun, pos) in nltk.pos_tag(tokens) if self.is_noun(pos)]

	def get_tokens(self, content: str):
		tokens = self.denoise(self.tokenize(content))

		return tokens

