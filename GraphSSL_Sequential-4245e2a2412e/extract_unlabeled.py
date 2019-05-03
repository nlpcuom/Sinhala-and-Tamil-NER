import sys
import numpy as np

unlabeled_file = open(sys.argv[1])
ann_loc = sys.argv[2]

suffixes = ['dev', 'test', 'train']

vocab = dict()

for suf in suffixes:
    file = open("%s%s.conllu" % (ann_loc, suf))
    for line in file:
        line = line.strip()
        if '#' not in line and line != '':
            word = line.split()[1]
            vocab[word] = 1

freq = []
sentences = []
sentence = ''
count = 0

for line in unlabeled_file:
    line = line.strip()
    if(line != ''):
        sentence += line + ' '
        if(line in vocab):
            count += 1
    else:
        sentences.append(sentence)
        freq.append(count)
        sentence = ''
        count = 0
args = np.argsort(freq)

counter = 0
for i in reversed(range(0, len(args))):
    if counter >= 500000:
        break
    for word in sentences[args[i]].split():
        print(word)
        counter += 1
    print('')
