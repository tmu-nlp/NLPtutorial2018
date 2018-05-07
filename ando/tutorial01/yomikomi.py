import math

dic = {}
wordcount = 0
partOfDic = 0
cul=0

f = open("model.txt").read()
f = f.rstrip()
words = f.split("\t")
for word in words:
    word = word.split(":")
    dic[word[0]] = word[1]

f = open("/Users/one/nlptutorial/data/wiki-en-test.word")
lines = f.readlines()
for line in lines:
    partcul = 1
    line = line.rstrip()
    test_words = line.split(" ")
    test_words.append("</s>")
    wordcount += len(test_words)
    for i in test_words:
        if i in dic:
            partcul *= 0.95*(float(dic[i]))+0.05*(1/1000000)
            partOfDic += 1
        else:
            partcul *= 0.05*(1/1000000)
    cul += math.log(partcul,2)
print("entropy="+str(-(cul/wordcount)))
print("coverage="+str(partOfDic/wordcount))
