from maml import MAML


num_way = 5
num_inner_steps = 1
inner_lr =.4
learn_inner_lrs = False
outer_lr = 0.001

source_dir = './logs/maml/omniglot.way:5.support:1.query:15.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16.prune:True'
same_dir = './logs/maml/lottery_same'
dif_dir = './logs/maml/lottery_dif'
source_state = 11000

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