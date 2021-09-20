files = ['train', 'dev', 'test']

for file in files:
    f_out = open('sst_data/{}.txt'.format(file), 'r')
    f_in = open('{}.txt'.format(file), 'w')
    text = f_out.readlines()
    for line in text:
        label, txt = line.strip().split('\t')
        f_in.write(txt + '\t' + label + '\n')
    f_out.close()
    f_in.close()
