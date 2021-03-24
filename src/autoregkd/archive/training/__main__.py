import click


def train_distilbart():
    from .train_distilbart_seq2seq import main

    main()


def train_distilbart_qa():
    from .train_distilbart_qa import main

    main()

