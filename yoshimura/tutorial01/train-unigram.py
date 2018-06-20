import sys
from collections import defaultdict

counts = defaultdict(lambda: 0)
total_count = 0

with open(sys.argv[1],"r") as train_file:
    for line in train_file:
        line = line.strip()
        words = line.split(" ")
        words.append("</s>")

        for word in words:
            counts[word] += 1 
            total_count += 1

with open('model',"w") as model_file:
    for word, count in sorted(counts.items()):
        probability = count / total_count
        model_file.write(word + "\t" + str(probability) + "\n")


# unknown : 496
# W : 4734

# entropy = 10.527337238682652
# coverage = 0.895226024503591
