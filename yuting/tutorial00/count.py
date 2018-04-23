import sys
my_file = open(sys.argv[1],"r")
data = my_file.read()
words = {}
 
for word in data.split():
 	words[word] = words.get(word, 0) + 1

d = [(v,k) for k,v in words.items()]
d.sort()
d.reverse()
for count, word in d [:20]:
	print(count, word)