import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

"""
To install spacy Languages use:
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
"""

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), batch_size=64, device="cuda"
)

for batch in train_iterator:
    print(batch)

print(english.vocab.stoi['the'])
print(english.vocab.stoi['i'])
