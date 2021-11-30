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

maml = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        save_dir+'0'
    )

maml._save(0)
prev_iteration_parameters = maml._meta_parameters
for i in range(lottery_iterations):
    sparisty = 1 - alpha**i
    mask = pruning.global_magnitude_pruning(prev_iteration_parameters, sparisty)
    maml._log_dir = save_dir+'0'
    maml.load(0)
    maml.set_mask(mask)
    maml._log_dir = save_dir+f'{i}'
    maml.reset_optimizer()
    writer = tensorboard.SummaryWriter(log_dir=save_dir+f'{i}')
    print('Starting to train model', i, 'with sparsity:')
    maml.print_sparisty()
    maml.train(
        dataloader_train,
        dataloader_val,
        writer, 
        False
    )
    print('Finished training model', i)
    prev_iteration_parameters = maml._meta_parameters
    