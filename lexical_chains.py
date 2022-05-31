from preprocessor import PreProcessor
from nltk.corpus import wordnet as wn

class Chain:
    def __init__(self, terms=[], synsets=[]):
        self.terms = terms
        self.synsets = synsets

    def get_terms(self):
        return self.terms

    def get_synsets(self):
        return self.synsets


class LexicalChains:
	def __init__(self):
		self.lexical_chains = []
		self.term_dict = {}

		self.WUP_SCORE_THRESHOLD = 0.7
		self.preprocessor = PreProcessor()

	def construct_chains(self, content: str):
		tokens = self.preprocessor.get_tokens(content)
		nouns = self.preprocessor.extract_nouns(tokens)

		# construct chains here
		for term in nouns:
			# only consider unique terms
			if term in self.term_dict:
				self.term_dict[term] += 1
				continue

			self.term_dict[term] = 1
			# PoS Tag = Noun
			term_synsets = wn.synsets(term, pos="n")
			# if no synsets for a term, consider it as new chain
			if len(term_synsets) == 0:
				self.lexical_chains.append(Chain(terms=[term]))
				continue

			need_new_chain = True
			for chain in self.lexical_chains:
				for synset in term_synsets:
					# if even one synset passes wup score threshold
					# then add it and move to next chain
					if not need_new_chain:
						break

					for chain_synset in chain.synsets:
						score = synset.wup_similarity(chain_synset)
						if score >= self.WUP_SCORE_THRESHOLD:
							chain.terms.append(term)
							chain.synsets += term_synsets
							need_new_chain = False
							break
			
			if need_new_chain:
				self.lexical_chains.append(Chain(terms=[term], synsets=term_synsets))

		return self.lexical_chains

	def show_chains(self):
		chain_index = 1
		for chain in self.lexical_chains:
			print(f"Chain {chain_index}: " + ", ".join([term for term in chain.get_terms()]))
			chain_index += 1
	
	def get_features(self):
		return list(self.term_dict.keys())

	def get_term_dict(self):
		return self.term_dict

# lexical_chains = []
# features = []
# terms_dict = {}

# WUP_SCORE_THRESHOLD = 0.7



# doc2 = p.get_text(f.read_file(
#     "http_^^cs.cornell.edu^Info^Courses^Spring-96^CS432^cs432.html"))

# tokens1 = nltk.word_tokenize(doc1)
# tokens2 = nltk.word_tokenize(doc2)


# def is_noun(pos): return pos[:2] == "NN"


# # select candidate words (nouns)
# nouns = [n for (n, pos) in nltk.pos_tag(tokens1)
#          if is_noun(pos) and n.isalpha()]


# class Chain():
#     def __init__(self, words=[], senses=[]):
#         self.words = words
#         self.senses = senses

#     def get_words(self):
#         return self.words

#     def get_senses(self):
#         return self.senses


# for term in nouns:
#     if term in terms_dict:
#         continue

#     terms_dict[term] = 1
#     term_synsets = wn.synsets(term, pos="n")
#     if len(term_synsets) == 0:
#         lexical_chains.append(Chain(words=[term]))
#         continue

#     need_new_chain = True
#     for chain in lexical_chains:
#         for synset in term_synsets:
#             if not need_new_chain:
#                 break

#             for sense in chain.senses:
#                 score = synset.wup_similarity(sense)
#                 if score >= WUP_SCORE_THRESHOLD:
#                     chain.words.append(term)
#                     chain.senses += term_synsets
#                     need_new_chain = False
#                     break

#     if need_new_chain:
#         lexical_chains.append(Chain(words=[term], senses=term_synsets))

# index = 1
# total_words = 0
# for chain in lexical_chains:
#     print("Chain "+str(index)+": "+", ".join(str(word)
#           for word in chain.get_words()))
#     index += 1
#     total_words += len(chain.get_words())

if __name__ == "__main__":
	from fetcher import Fetcher
	from parser import Parser
	f = Fetcher()
	p = Parser()

	f.cd_dir("course-cotrain-data/fulltext/course")
	doc1 = p.get_text(f.read_file("http_^^www.ece.wisc.edu^~jes^ece752.html"))
	
	l = LexicalChains()
	l.construct_chains(doc1)
	l.show_chains()
