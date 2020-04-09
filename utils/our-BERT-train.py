import os

res = open("datasets/dev-SI.txt", 'r')

lines = res.readlines()
new = open("new-dev-si.txt", "w")

remove_short = True

word_list = []
for line in lines:
    words, labels, _, _ = line.split("\t")
    if
    for word, label in zip(words.split(), labels.split()):
        word_list.append(word + " " + label + "\n")
    word_list.append("\n")

new.writelines(word_list)
