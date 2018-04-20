from collections import defaultdict

def word_count(text):
  word_dict = defaultdict(int)
  for line in text:
    words = line.split()
    for word in words:
      word_dict[word] += 1
  return word_dict

if __name__ == '__main__':
  text = open('../../data/wiki-en-train.word').readlines()
  for word, count in sorted(word_count(text).items(), key = lambda x: -x[1]):
    print(word, count)
  print('単語の異なり数: {}'.format(len(word_count(text))))
