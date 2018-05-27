from collections import defaultdict
import re

def train_unigram(input_file_path, model_file_path):

	counts = defaultdict(lambda: 0)
	total_count = 0
	model_result = {}

	with open(input_file_path, 'r') as input_file:
		for line in input_file:
			line = line.strip()
			line = re.sub(r'　 ', '', line)
			words = line.split(" ")
			words.append("</s>")
			for word in words:
				counts[word] += 1
				total_count += 1

	with open(model_file_path, 'w') as model_file:
		for word, count in counts.items():
			probability = float(count) / total_count
			model_result[word] = probability
			print('{}\t{}'.format(word, probability), file=model_file)

