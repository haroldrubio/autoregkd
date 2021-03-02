import click

@click.command(
    context_settings=dict(show_default=True),
)

@click.option(
    "--log_batch_size",
    type=int,
    default=1,
    help="batch size for training will be 2**LOG_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--log_eval_batch_size",
    type=int,
    default=1,
    help="batch size for eval will be 2**LOG_EVAL_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--epochs",
    type=int,
    default=1,
    help="number of training epochs",
)
@click.option(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="learning rate",
)

def experiment(**config):
    """Train a BART model"""
    from ..training.train import training
    
    training(config)
