from bs4 import BeautifulSoup

class Parser:
	def __init__(self):
		self.bs4_instance = BeautifulSoup

	def parse(self, content: str, parser="html.parser"):
		return self.bs4_instance(content, parser)

	def get_text(self, content: str, parser="html.parser", return_as_tokens=False):
		soup = self.parse(content)
		
		if return_as_tokens:
			return [token for token in soup.stripped_strings]

		return soup.get_text()


if __name__ == "__main__":
	from fetcher import Fetcher
	print("test")
	f = Fetcher()
	f.cd_dir("course-cotrain-data/fulltext/course")
	p = Parser()
	q = p.parse(f.read_file("http_^^www.ece.wisc.edu^~jes^ece752.html"))
	r = p.parse(f.read_file("http_^^cs.cornell.edu^Info^Courses^Spring-96^CS432^cs432.html")).get_text()
	print([t for t in q.stripped_strings])
	input()
	print("\n\n\n")
	print(r)
