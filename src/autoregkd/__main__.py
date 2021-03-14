import click
from .training import train_distilbart_seq2seq
from .training import train_distilbart_qa


@click.group()
def main():
    pass


main.add_command(train_distilbart_seq2seq, "train_distilbart_seq2seq")
main.add_command(train_distilbart_qa, "train_distilbart_qa")
