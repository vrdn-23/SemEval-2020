import os

DEV_ARTICLES_DIR = "datasets/dev-articles/"

res = open("datasets/unlabelled-dev-SI.txt", 'w')

for filename in os.listdir(DEV_ARTICLES_DIR):
    with open(DEV_ARTICLES_DIR + filename, 'r') as f:
        offset = 0
        for line in f:
            line = line.replace(chr(8212), " ")
            line = line.replace(chr(225), "a")
            line = line.replace(chr(233), "e")
            line = line.replace(chr(195), "A")
            line = line.replace("-", " ")
            line = line.replace("=", " ")
            sentence_offset = offset
            labels = []
            words = line.rstrip().split(" ")
            if len(words) == 1 and words[0] == '':
                offset += 1
                continue
            for word in words:
                if not word:
                    offset += 1
                    continue
                word_starting_off = offset
                word_ending_off = offset + len(word) - 1
                offset += len(word) + 1

            res.write("\t".join([" ".join(words), filename, str(sentence_offset)]))
            res.write("\n")

res.close()
print("Done.")
