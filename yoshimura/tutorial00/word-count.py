# coding:utf-8
import sys

dict = {}
file = open(sys.argv[1],"r")

for line in file:
    line = line.strip()
    words = line.split(" ")
    
    for word in words:
        if word not in dict:
            dict[word] = 1
        else:
            dict[word] += 1
        
for key, value in sorted(dict.items()):
    print(f"{key} {value}")

print(f"異なり数（単語タイプ数）: {len(dict)} ")

