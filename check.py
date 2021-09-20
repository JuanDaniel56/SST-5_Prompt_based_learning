from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('./pretrained_model')

t = ['terrible', 'bad', 'common', 'good', 'best']
#s = "it is terrible"
print(t)
print(tokenizer.convert_tokens_to_ids(t))

for token in t:
    s = 'it is ' + token
    print(s)
    print('encode')
    print(tokenizer.encode(s, add_special_tokens = False))
