dic={}
total=0
token=0

f=("")
lines=f.readlines()
for line in lines:
	line=line.rstrip()
	words=line.split("")
	words.append("</s")
	for word in words:
		if word in dic:
			dic[word]+=1
		else:
			dic[word]=1
			total+=1

		token+=1

for i,j in dic.items():
	with open("model.txt","a")as fout:
		fout.write("{}:{}\t".format(i,j/token))

print(total)