import math

dic = {}
dic2gram = {}
wordcount = 0
cul=0

f = open("model.txt").read()
f = f.rstrip()
words = f.split("\t")
for word in words:
    word = word.split(":")
    dic[word[0]] = word[1]
f = open("model2.txt").read()
f = f.rstrip()
words = f.split("\t")
for word in words:
    word = word.split(":")
    if word[0] not in dic2gram:
        dic2gram[word[0]] = {}
        dic2gram[word[0]][word[1]]=float(word[2])


f = open("/Users/one/nlptutorial/data/wiki-en-test.word")
lines = f.readlines()
for line in lines:
    partcul = 1
    line = line.rstrip()
    test_words = line.split(" ")
    test_words.append("</s>")
    test_words.insert(0, “<s>”) 
    wordcount += len(test_words)
    for i in test_words:
        if i != “<s>”:
            if i in dic:
                one = 0.95*(float(dic[i]))+0.05*(1/1000000)
            else:
                one = 0.05*(1/1000000)
            if i in dic2gram:
                if lastword in dic2gram[i]:
                    partcul *= 0.95*(float(dic2gram[i][lastword]))+0.05*one
                else:
                    partcul *= 0.05*one
    cul += math.log(partcul,2)
print("entropy="+str(-(cul/wordcount)))
