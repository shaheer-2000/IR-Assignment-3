from main import main as get_model
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tfidf = True
ttc = True
lexchains = True
classifier, vectorizer = get_model(use_tfidf=tfidf, use_ttc=ttc, use_lexchains=lexchains, debug=True)

@app.route('/', methods=['POST'])
def main():
	global classifier, lexchains
	document = request.json["document"]
	vector = vectorizer.vectorize(document, is_binary=lexchains)

	result = classifier.predict(vector)
	if result == 0:
		label = "Non-Course"
	else:
		label = "Course"

	return { "label": label }
	

@app.route('/settings', methods=['POST'])
def settings():
	global classifier, vectorizer, tfidf, ttc, lexchains
	settings = request.json["settings"]

	tfidf, ttc, lexchains = tuple(map(lambda x: int(x) == 1, settings.split("-")))
	classifier, vectorizer = get_model(use_tfidf=tfidf, use_ttc=ttc, use_lexchains=lexchains, debug=True)

	return { "success": True }


if __name__ == "__main__":
	app.run(debug=True)