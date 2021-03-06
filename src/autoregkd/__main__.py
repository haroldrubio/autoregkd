import click
from .training import train_distilbart


@click.group()
def main():
    pass


main.add_command(train_distilbart, "train_distilbart")
