from pathlib import Path

class Fetcher:
	def __init__(self):
		self.curr_dir = Path.cwd()

	def get_dir(self):
		return self.curr_dir

	def cd_dir(self, target_dir):
		target_dir_path = Path(target_dir)
		if not target_dir_path.exists() or not target_dir_path.is_dir():
			raise FileNotFoundError(f"{target_dir} is not a valid directory path")

		self.curr_dir = target_dir_path
	
	def get_files(self, filter_fn=lambda f: True):
		"""Get all the files in the current directory

		Args:
			filter_fn (function): a callback function that evaluates to a boolean value

		Returns:
			List[str]: a list containing the relative paths to each file
		"""
		return [file for file in self.curr_dir.iterdir() if filter_fn(file)]

	def read_file(self, file: str | Path, encoding="utf-8"):
		"""_summary_

		Args:
			file (str | Path): name of the target file or the Path object for it
			encoding (str, optional): the encoding to open the file in. Defaults to "utf-8".

		Raises:
			FileNotFoundError: if the file could not be found, raises this exception

		Returns:
			str | None: returns the content of the file if it is read, else returns None
		"""
		file_path: str | Path = file
		if type(file) is str:
			file_path: Path = self.curr_dir / file

		if not file_path.exists() or not file_path.is_file():
			raise FileNotFoundError(f"{file} does not exist")

		try:
			return file_path.read_text(encoding=encoding, errors="replace")
		except Exception as e:
			print(e)

		return None


if __name__ == "__main__":
	fetcher = Fetcher()
	fetcher.cd_dir("course-cotrain-data/fulltext/course")
	print(fetcher.get_dir())
	files = fetcher.get_files()
	for file in files:
		text = fetcher.read_file("http_^^www.ece.wisc.edu^~jes^ece752.html")
		print(text)
		input()

