import sys
my_file = open(sys.argv[1], "r")

dictionary = dict()
for line in my_file:
    line = line.strip()
    words = line.split(" ")
    for word in words:
        if word in dictionary.keys():
            dictionary[word] += 1
        else:
            dictionary[word] = 1

for key, value in sorted(dictionary.items()):
    print(f'{key}\t{value}')