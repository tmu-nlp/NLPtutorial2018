dic={}
dic2gram={}
total=0
token=0
#lastword=“”

f = open("/Users/one/nlptutorial/data/wiki-en-test.word")
lines = f.readlines()
for line in lines:
    line = line.rstrip()
    words = line.split(" ")
    words.append("</s>")
    words.insert(0,"<s>") 
    for word in words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1
            total += 1
        token+=1
        if word != "<s>":
            if lastword not in dic2gram:
                dic2gram[lastword] = {}
            else:
                cont = dic2gram[lastword]
                if word not in cont:
                    cont[word] = 1
                else:
                    cont[word] += 1
        lastword = word

with open("model.txt", "w") as fout:
    for i,j in dic.items():
        fout.write("{} {}\t".format(i,j/token))
with open("model2.txt", "w") as fout:
    for i in dic2gram:
        for j in dic2gram[i]:
            num = dic2gram[i][j]
            dic2gram[i][j] = num/dic[i]        
            fout.write("{} {} {}\t".format(i,j,dic2gram[i][j]))
