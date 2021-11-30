from maml import MAML
import pruning
import omniglot
from torch.utils import tensorboard

alpha = .8
lottery_iterations = 15
num_way = 5
num_inner_steps = 1
inner_lr =.4
learn_inner_lrs = True
outer_lr = 0.001
batch_size = 16
num_support = 1
num_query = 15
num_train_iterations = 4000
sparsities = []
save_dir = './logs/maml/lottery/'

num_training_tasks = batch_size * num_train_iterations

dataloader_train = omniglot.get_omniglot_dataloader(
    'train',
    batch_size,
    num_way,
    num_support,
    num_query,
    num_training_tasks
)
dataloader_val = omniglot.get_omniglot_dataloader(
    'val',
    batch_size,
    num_way,
    num_support,
    num_query,
    batch_size * 4
)
for i in [2, 4, 6, 8, 10, 12, 14]:
    maml_mask = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        save_dir+f'{i}'
    )
    maml_train = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        save_dir+f'{i}_cntrl'
    )
    maml_mask.load(100)
    maml_train.set_mask(maml_mask._mask)
    maml_train.reset_optimizer()
    writer = tensorboard.SummaryWriter(log_dir=save_dir+f'{i}_cntrl')
    print('Starting to train model', i, 'with sparsity:')
    maml_train.print_sparisty()
    maml_train.train(
        dataloader_traind,
        dataloader_val,
        writer, 
        False
    )

