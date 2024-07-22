import numpy as np
import torch
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = datetime.now()
dt_string = start_time.strftime("%Y%m%d-%H.%M")

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set the default device to the first available GPU (index 0)
    print("CUDA available, using GPU")
    torch.cuda.set_device(0)
    DEVICE = torch.device('cuda:0')
else:
    print("WARNING: CUDA not available, using CPU")
    DEVICE = torch.device('cpu')


#########################################################
# ACTIVATION FUNCTIONS
#########################################################

relu = torch.nn.ReLU()
tanh = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()
silu = torch.nn.SiLU()
linear = torch.nn.Identity()
leaky_relu = torch.nn.LeakyReLU()

@torch.jit.script
def sigmoid_derivative(x):
    return torch.exp(-x)/((1.+torch.exp(-x))**2)

@torch.jit.script
def relu_derivative(x):
    return torch.heaviside(x, torch.tensor(0.))

@torch.jit.script
def tanh_derivative(x):
    return 1-tanh(x)**2

@torch.jit.script
def silu_derivative(x):
    return silu(x) + torch.sigmoid(x)*(1.0-silu(x))

@torch.jit.script
def leaky_relu_derivative(x):
    return torch.where(x > 0, torch.tensor(1.), torch.tensor(0.01))

def get_derivative(f):
    if f == sigmoid:
        return sigmoid_derivative
    elif f == relu:
        return relu_derivative
    elif f == tanh:
        return tanh_derivative
    elif f == silu:
        return silu_derivative
    elif f == linear:
        return 1
    elif f == leaky_relu:
        return leaky_relu_derivative
    else:
        raise NotImplementedError(f"Derivative of {f} not implemented")


#########################################################
# GENERAL METHODS
#########################################################

def onehot(y_batch, N):
    """
    y_batch: tensor of shape (batch_size, 1)
    N: number of classes
    """
    return torch.eye(N, device=DEVICE)[y_batch.squeeze().long()].float()

def to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()

def preprocess_batch(batch):
    batch[0] = set_tensor(batch[0])
    batch[1] = set_tensor(batch[1])
    return (batch[0], batch[1])

def preprocess(dataloader):
    return list(map(preprocess_batch, dataloader))

def set_tensor(tensor):
    return tensor.to(DEVICE)

def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_parameters(layers, use_bias):
    """
    Counts the number of parameters in a hierarchical network with given layer arrangement.
    """
    n_params = 0
    for i in range(len(layers)-1):
        n_params += layers[i]*layers[i+1]
        if use_bias:
            n_params += layers[i+1]
    return n_params

#########################################################
# LITTLE DATA METHODS
#########################################################

def train_subset_indices(train_set, n_classes, no_per_class):
    """
    Selects indices of a subset of the training set, with no_per_class samples per
    class. Useful for training with less data.
    """
    if no_per_class ==0:  # return all indices
        return np.arange(len(train_set))
    else:
        train_indices = []
        for i in range(n_classes):
            train_targets = torch.tensor([train_set.dataset.targets[i] for i in train_set.indices]) # SLOW but fine for now
            indices = np.where(train_targets == i)[0]
            indices = np.random.choice(indices, size=no_per_class, replace=False)
            train_indices += list(indices)
        return train_indices

def print_class_counts(loader):
    """
    Prints the number of samples per class in the given dataloader.
    """
    n_classes = loader.dataset.dataset.targets.unique().shape[0]
    counts = torch.zeros(n_classes)
    for _, (_, y_batch) in enumerate(loader):
        for i in range(n_classes):
            temp = (torch.argmax(y_batch,dim=1) == i)
            counts[i] += torch.count_nonzero(temp) 
    print(f"Class counts: {counts}")


#########################################################
# PCG METHODS
#########################################################

def get_mask_hierarchical(layers, symmetric=False):
    """
    Generates a hierarchical mask for the given layer arrangement.
    Returns:
        torch.Tensor: A binary mask matrix of shape (N, N) where N is the total number of nodes.
    """
    rows, cols = get_mask_indices_hierarchical(layers)
    N = np.sum(layers)
    M = torch.zeros((N, N), device=DEVICE)
    M[rows, cols] = torch.ones(len(rows), device=DEVICE)
    M = M.T
    if symmetric:
        # Make the matrix symmetric
        M = torch.tril(M) + torch.tril(M).T
    return M

def get_nodes_partition(layers):
    """
    Partitions nodes into layers based on the layer arrangement.
    Returns:
        list[list[int]]: A list of lists, where each sublist contains the nodes of a layer.
    """
    nodes = np.arange(sum(layers))  # Number the nodes 0, ..., N-1

    nodes_partition = []
    for i in range(len(layers)):
        a = np.sum(layers[:i]).astype(int)
        b = np.sum(layers[:i+1]).astype(int)
        nodes_partition.append(nodes[a:b])
    return nodes_partition 

def get_mask_indices_hierarchical(layers):
    """
    Finds the matrix indices of nonzero weights for a hierarchical mask.
    Returns:
        tuple[list[int], list[int]]: Two lists representing the row and column indices of the nonzero weights.
    """
    # Partition nodes into layers
    nodes_partition = get_nodes_partition(layers)

    # Helper function to combine rows and columns
    def combine(x, y):
        z = np.array([(x_i, y_j) for x_i in x for y_j in y])
        rows, cols = z.T
        return rows.tolist(), cols.tolist()  # Returns rows, cols of matrix indices

    # Find matrix indices of nonzero weights
    all_rows, all_cols = [], []
    for i in range(len(layers)-1):
        rows, cols = combine(nodes_partition[i], nodes_partition[i+1])
        all_rows += rows
        all_cols += cols

    return all_rows, all_cols

