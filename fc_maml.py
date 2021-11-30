
import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd  # pylint: disable=unused-import
from torch.utils import tensorboard
from tqdm import tqdm
import omniglot
import miniimagenet
import util  # pylint: disable=unused-import
import pruning 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 250
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600
LAYER_SIZES = [256, 128, 64, 64]
class fc_MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir, 
            input_size = 784,
            layer_sizes = LAYER_SIZES
    ):
        """Inits MAML.

        The network consists of fully connect layers
        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
        """
        meta_parameters = {}

        # construct feature extractor
        self._n_layers = len(layer_sizes) + 1
        total_layer_sizes = [input_size]+ layer_sizes + [num_outputs]
        self._input_size = input_size
        for i in range(self._n_layers):
        # construct linear head layer
            meta_parameters[f'w{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    total_layer_sizes[i+1],
                    total_layer_sizes[i],
                    requires_grad=True,
                    device=DEVICE
                )
            )
            meta_parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(
                    total_layer_sizes[i+1],
                    requires_grad=True,
                    device=DEVICE
                )   
            )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self._mask = {key: torch.ones(value.shape, requires_grad=False, device=DEVICE) for key, value in meta_parameters.items()}

    def _forward(self, images, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        num_images = images.shape[0]
        x = images.reshape(num_images, -1)
        for i in range(self._n_layers - 1):
            x = F.linear(input=x,
                weight=parameters[f'w{i}']*self._mask[f'w{i}'],
                bias=parameters[f'b{i}']*self._mask[f'b{i}']
                )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        return F.linear(
            input=x,
            weight=parameters[f'w{self._n_layers - 1}']*self._mask[f'w{self._n_layers - 1}'],
            bias=parameters[f'b{self._n_layers - 1}']*self._mask[f'b{self._n_layers - 1}']
        )

    def _inner_loop(self, images, labels, train):   # pylint: disable=unused-argument
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating
        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        for _ in range(self._num_inner_steps):
            output = self._forward(images, parameters)
            accuracies.append(util.score(output, labels))
            loss = F.cross_entropy(output, labels)
            gradients = torch.autograd.grad(loss, parameters.values(), retain_graph = True, create_graph = train)
            for key, grad in zip(parameters.keys(), gradients):
                parameters[key] = parameters[key].clone() - self._inner_lrs[key]* grad
        output = self._forward(images, parameters)
        accuracies.append(util.score(output, labels))  

        return parameters, accuracies

    def _outer_step(self, task_batch, train): 
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)

            parameters, accuracies = self._inner_loop(images_support, labels_support, train)
            output = self._forward(images_query, parameters)
            accuracy_query_batch.append(util.score(output, labels_query))
            outer_loss_batch.append(F.cross_entropy(output, labels_query))
            accuracies_support_batch.append(accuracies)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def test_pruning(self, dataloader_test): 
        n_dif = 12
        compound_density = [.8**i for i in range(n_dif)]
        cum_accuracies = [[] for _ in range(n_dif)]
        for task_batch in tqdm(dataloader_test):
            for task in task_batch:
                images_support, labels_support, images_query, labels_query = task
                images_support = images_support.to(DEVICE)
                labels_support = labels_support.to(DEVICE)
                images_query = images_query.to(DEVICE)
                labels_query = labels_query.to(DEVICE)

                parameters, _ = self._inner_loop(images_support, labels_support, False)
                base_sparsity = self.get_sparsity()
                for i in range(n_dif):
                    mask = pruning.global_magnitude_pruning(parameters, 1 -compound_density[i]*(1 - base_sparsity))
                    masked_parameters = {
                        k: torch.clone(v)*mask[k]
                        for k, v in parameters.items()
                    }
                    output = self._forward(images_query, masked_parameters)
                    cum_accuracies[i].append(util.score(output, labels_query))

        mean = [np.mean(cum_accuracies[i]) for i in range(n_dif)]
        err_95ci = [1.96 *np.std(cum_accuracies[i])/np.sqrt(len(cum_accuracies[i])) for i in range(n_dif)]

        print('means: ', mean)
        print('err_95ci: ', err_95ci)
        


    def train(self, dataloader_train, dataloader_val, writer, prune):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        PREPRUNE_STEPS = 15000
        PRUNE_EVERY = 5000
        PRUNE_FRAC = .8
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()


            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

            if prune and i_step >= PREPRUNE_STEPS and i_step%PRUNE_EVERY == 0:
                next_sparsity =  1 - PRUNE_FRAC**((i_step - PREPRUNE_STEPS)/PRUNE_EVERY + 1)
                self.set_mask(pruning.global_magnitude_pruning(self._meta_parameters, next_sparsity))
                print(f'PRUNING to {next_sparsity} sparsity')

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in tqdm(dataloader_test):
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
        return mean, mean_95_confidence_interval
    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path, map_location=torch.device(DEVICE)) #i changed this line
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            if 'mask' in state:
                self._mask = state['mask']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def set_mask(self, mask):
        self._mask = mask
        with torch.no_grad():
            for k in self._meta_parameters.keys():
                self._meta_parameters[k]*=mask[k]
        self.reset_optimizer()
    def get_sparsity(self):
        return 1 - torch.cat([tensor.view(-1) for tensor in self._mask.values()]).mean()

    def print_sparisty(self):
        print('lr:', self._inner_lrs)
        print('avg_sparsity: ', self.get_sparsity()) 
        for key, value in self._mask.items():
            print(key, 1 - value.mean())
        
    def reset_optimizer(self):
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 mask = self._mask,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/fc_maml/layers.{"-". join([str(i) for i in LAYER_SIZES])}.omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.batch_size:{args.batch_size}.prune:{args.prune}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    gen_dataset_function = miniimagenet.get_miniimagenet_dataloader if args.mini_imagenet else omniglot.get_omniglot_dataloader
    maml = fc_MAML(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = gen_dataset_function(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = gen_dataset_function(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        maml.train(
            dataloader_train,
            dataloader_val,
            writer, 
            args.prune
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        maml.print_sparisty()
        dataloader_test = gen_dataset_function(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        if args.prune:
            maml.test_pruning(dataloader_test)
        else:
            _, _ = maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a (fc) MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=50000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--prune', default=False, action='store_true',
                        help=('prune while training?'))
    parser.add_argument('--mini_imagenet', default=False, action='store_true',
                        help=('use miniimagenet instead of omniglot'))
    main_args = parser.parse_args()
    main(main_args)