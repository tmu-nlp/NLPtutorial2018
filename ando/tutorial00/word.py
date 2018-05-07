import re

dic = {}
num = 0
h = open("/Users/one/nlptutorial/data/wiki-en-train.word")
lines = h.readlines()
for line in lines:
    line = line.rstrip()
    line.replace(" .","").replace(" ,","")
    line = line.split(" ")
    for i in line:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
        judge = re.match('[0-9]+$',i)
        if not isinstance(judge,type(None)):
            num += 1
for i,j in dic.items():
    print(i,j)
print(len(dic))
print(num)
