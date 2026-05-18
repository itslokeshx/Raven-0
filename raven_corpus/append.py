import sys
outfile = open('/home/loki/code/Raven/raven/data/corpus.txt', 'a', encoding='utf-8')
for f in sys.argv[1:]:
    with open(f, 'r', encoding='utf-8', errors='ignore') as infile:
        outfile.write('\n' + infile.read() + '\n')
outfile.close()
