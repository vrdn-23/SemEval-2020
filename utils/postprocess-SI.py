SUBMISSION_FILENAME = "datasets/submission-SI.txt"
MODEL_OUTPUT_FILENAME = "datasets/labelled_test-SI.txt"

with open(MODEL_OUTPUT_FILENAME, 'r') as f:
    lines = f.readlines()

buffer = ""
for line in lines:
    sen, label, name, offset = line.rstrip().split('\t')
    name = name.split('.')[0][7:]
    sen = sen.rstrip().split(' ')
    label = label.rstrip().split(' ')
    start_offset = int(offset)
    end_offset = int(start_offset)
    for t, (i, j) in enumerate(zip(sen, label)):
        if int(j) == 1:
            end_offset += (len(i) + 1)
        if int(j) == 0:
            if end_offset != start_offset:
                buffer += '\t'.join([name, str(start_offset), str(end_offset - 1)])
                # buffer += '\t'.join([name, str(start_offset), str(end_offset - 1), sen[t-1]])
                buffer += '\n'
                start_offset = end_offset
            start_offset += (len(i) + 1)
            end_offset = start_offset
    if end_offset != start_offset:
        buffer += '\t'.join([name, str(start_offset), str(end_offset - 1)])
        # buffer += '\t'.join([name, str(start_offset), str(end_offset - 1), sen[-1]])
        buffer += '\n'

with open(SUBMISSION_FILENAME, 'w') as f:
    f.write(buffer)

print("Done.")
