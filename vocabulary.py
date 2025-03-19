from datasets import load_dataset
from datasets.table import Table
from nltk.tokenize import word_tokenize


class Vocabulary:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []

    def build_vocabulary(self):
        ds = load_dataset("wmt/wmt14", "de-en")
        trainTable: Table = ds.data["train"]
        train = trainTable.to_pylist()
        
        self.add_word("<sos>")
        self.add_word("<eos>")
        for train_data in train:
            en = train_data["translation"]["en"]
            de = train_data["translation"]["de"]
            words_en = word_tokenize(en)
            words_de = word_tokenize(de)

            print(words_en)
            print(words_de)

            for word in words_en:
                self.add_word(word)
            for word in words_de:
                self.add_word(word)
            break


    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)

    def __len__(self):
        return len(self.index_to_word)


if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.build_vocabulary()
    print(vocab.word_to_index)
    print(vocab.index_to_word)
    # print(len(vocab))
