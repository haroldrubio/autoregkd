import click
from .training import run_seq2seq
from .training import run_qa


@click.group()
def main():
    pass


main.add_command(run_seq2seq, "run_seq2seq")
main.add_command(run_qa, "run_qa")
