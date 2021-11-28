from maml import MAML
import pruning

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
save_dir = './logs/maml/lottery_omniglot.way:5.support:1.query:15.inner_steps:1.inner_lr:0.4.learn_inner_lrs:True.outer_lr:0.001.batch_size:16'

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
    args.batch_size,
    args.num_way,
    args.num_support,
    args.num_query,
    args.batch_size * 4
)



maml_mask = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        source_dir
    )
maml_mask.load(source_state)

maml_same_init = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        same_dir
    )
maml_same_init._log_dir = source_dir
maml_same_init.load(0)
maml_same_init._log_dir = same_dir

maml_new_init = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        dif_dir
    )

maml_same_init.set_mask(maml_mask._mask)
maml_new_init.set_mask(maml_mask._mask)

maml_same_init._save(0)
maml_new_init._save(0)