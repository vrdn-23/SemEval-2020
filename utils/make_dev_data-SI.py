import os

DEV_ARTICLES_DIR = "datasets/dev-articles/"

sentences = []
counter = 0
for filename in os.listdir(DEV_ARTICLES_DIR):
    counter += 1
    with open(DEV_ARTICLES_DIR + filename, 'r') as f:
        sen = ""
        start = 0
        end_counter = 0
        c = f.read(1)
        sen += c
        while c != '':
            if c == ' ':
                end_counter = 0
            if c == '\n':
                if end_counter == 0:
                    sen = sen[:-1]
                    sentences.append('\t'.join([sen, filename, str(start - len(sen))]))
                    sen = ""
                    end_counter += 1
                else:
                    sen = sen[:-1]
            start += 1
            c = f.read(1)
            sen += c
        if sen != '':
            sentences.append('\t'.join([sen, filename, str(start - len(sen))]))

with open("datasets/dev-SI.txt", 'w') as f:
    f.writelines(["%s\n" % x for x in sentences])

print("Done.")
