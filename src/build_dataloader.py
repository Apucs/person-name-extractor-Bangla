from data.dataloader import Corpus


class BuildDataLoader:
    def __init__(self, w2v):
        """
        Build the dataloader for train, validation and test
        Args:
            w2v: int(0/1)
                Whether model pretrained word2vec is available or not
        """
        if w2v == 0:
            self.corpus = Corpus(
                input_folder="dataset/",

                # Creating loader with modified labels
                # train_data="train_up_ner.tsv",
                # val_data="test_up_ner.tsv",
                # test_data="test_up_ner.tsv",

                train_data="train_data_ner.tsv",
                val_data="test_data_ner.tsv",
                test_data="test_data_ner.tsv",
                min_word_freq=1,
                batch_size=64,
            )
        else:
            self.corpus = Corpus(
                input_folder="dataset/",

                # Creating loader with modified labels
                # train_data="train_up_ner.tsv",
                # val_data="test_up_ner.tsv",
                # test_data="test_up_ner.tsv",

                train_data="train_data_ner.tsv",
                val_data="test_data_ner.tsv",
                test_data="test_data_ner.tsv",
                min_word_freq=1,
                batch_size=64,
                w2v=1
            )

        self.train_iter = self.corpus.train_iter
        self.val_iter = self.corpus.val_iter
        self.test_iter = self.corpus.test_iter

        print(f"No of sentences in Train set: {len(self.corpus.train_dataset)} sentences")
        print(f"No of sentences in Val set: {len(self.corpus.val_dataset)} sentences")
        print(f"No of sentences in Test set: {len(self.corpus.test_dataset)} sentences")

        print("\nNo of batches in train dataset:", len(self.train_iter))
        print("No of batches in validation dataset:", len(self.val_iter))
        print("No of batches in test dataset:", len(self.test_iter))
