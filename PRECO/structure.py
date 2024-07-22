from PRECO.optim import *
from PRECO.utils import *
from PRECO.structure import *
from torch import nn
from scipy.ndimage import label, find_objects

class PCStructure:
    """
    Abstract class for PC structure.

    Args:
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
    """
    def __init__(self, f, use_bias):
        self.f = f
        self.dfdx = get_derivative(f)
        self.use_bias = use_bias

class PCmodel:
    """
    Abstract class for PC model.
    
    Args:
        structure (PCStructure): Structure of the model.
        lr_x (float): Learning rate for the input.
        T_train (int): Number of training iterations.
        incremental (bool): Whether to use incremental EM.
        min_delta (float): Minimum change in energy for early stopping.
        early_stop (bool): Whether to use early stopping.
    """
    def __init__(self, structure: PCStructure, lr_x: float, T_train: int, 
                 incremental: bool, min_delta: float, early_stop: bool):
        self.structure = structure
        self.lr_x = torch.tensor(lr_x, dtype=torch.float, device=DEVICE)
        self.T_train = T_train
        self.incremental = incremental
        self.min_delta = min_delta
        self.early_stop = early_stop

    def weight_init(self, param):
        nn.init.normal_(param, mean=0, std=0.05)   

    def bias_init(self, param):
        nn.init.normal_(param, mean=0, std=0) 


class PCGStructure(PCStructure):
    """
    Abstract class for PCG structure.

    Args:
        shape (tuple): Number of input, hidden, and output nodes.
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
        mask (torch.Tensor): Mask for the weight matrix.
    """
    def __init__(self, shape, f, use_bias, mask):
        super().__init__(f, use_bias)
        self.shape = shape
        self.mask = mask

        if self.mask is not None:
            if np.all(np.triu(mask, k=1) == 0):
                labeled_matrix, self.num_layers = label(mask, structure=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]]))
                blocks = np.array([(slice_obj[1].stop - slice_obj[1].start, 
                                  slice_obj[0].stop - slice_obj[0].start)
                                  for slice_obj in find_objects(labeled_matrix)])
                self.layers = blocks[:, 0].tolist() + [blocks[-1, -1]]
                logging.info(f"Hierarchical mask, layers: {self.num_layers}, using feedforward initialization and testing.")
            else:
                self.num_layers = None
                logging.info("Non-hierarchical mask, using random initialization and iterative testing.")
        self.N = sum(self.shape)

    @property
    def hparams(self):
        return {"shape": self.shape, "f": self.f, "use_bias": self.use_bias, "mask": self.mask}

    def pred(self, x, w, b):
        raise NotImplementedError

    def grad_x(self, x, e, w, b, train):
        raise NotImplementedError

    def grad_w(self, x, e, w, b):
        raise NotImplementedError

    def grad_b(self, x, e, w, b):
        raise NotImplementedError
    

class PCG_AMB(PCGStructure):
    """
    PCGStructure class with convention: mu = wf(x)+b.
    """
    def pred(self, x, w, b):
        bias = b if self.use_bias else 0
        return torch.matmul(self.f(x), w.T) + bias

    def grad_x(self, x, e, w, b, train):
        lower = self.shape[0]
        upper = -self.shape[2] if train else sum(self.shape)
        return e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)

    def grad_w(self, x, e, w, b):
        out = -torch.matmul(e.T, self.f(x))
        if self.mask is not None:
            out *= self.mask
        return out

    def grad_b(self, x, e, w, b,):
        return -torch.sum(e, axis=0)


class PCG_MBA(PCGStructure):
    """
    PCGStructure class with convention: mu = f(xw+b).
    """
    def pred(self, x, w, b):
        bias = b if self.use_bias else 0
        return self.f(torch.matmul(x, w.T) + bias)
    
    def grad_x(self, x, e, w, b, train):
        lower = self.shape[0]
        upper = -self.shape[2] if train else sum(self.shape)
        bias = b[lower:upper] if self.use_bias else 0
        temp = self.dfdx( torch.matmul(x, w.T)[:,lower:upper] + bias)
        return e[:,lower:upper] - temp*torch.matmul(e, w.T[lower:upper,:].T)
    
    def grad_w(self, x, e, w, b):
        bias = b if self.use_bias else 0
        temp = e*self.dfdx( torch.matmul(x, w.T) + bias )
        out = -torch.matmul( temp.T,  x ) # matmul takes care of batch sum
        out *= self.mask if self.mask is not None else 1
        return out

    def grad_b(self, x, e, w, b):
        bias = b if self.use_bias else 0
        temp = self.dfdx( torch.matmul(x, w.T) + bias )
        return torch.sum(-e*temp, axis=0) # batch sum



class PCNStructure(PCStructure):
    """
    Abstract class for PCN structure.

    Args:
        layers (list): Number of nodes in each layer.
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
        upward (bool): Whether the structure is upward (discriminative) or downward (generative).
        fL (torch.Tensor -> torch.Tensor): Activation function for the last layer.
    """
    def __init__(self, layers, f, use_bias, upward, fL=None):
        super().__init__(f, use_bias)
        self.layers = layers
        self.upward = upward
        if fL is None:
            self.fL = f
            self.dfLdx = self.dfdx
        else:
            self.fL = fL
            self.dfLdx = get_derivative(fL)
        self.L = len(layers) - 1

    @property
    def hparams(self):
        return {"layers": self.layers, "f": self.f, "use_bias": self.use_bias, "upward": self.upward}

    # NOTE: this implementation is somewhat inefficient; costs around 1s/epoch for MNIST
    def fl(self, x, l):
        if l == self.L:
            return self.fL(x)
        else:
            return self.f(x)
        
    def dfldx(self, x, l):
        if l == self.L:
            return self.dfLdx(x)
        else:
            return self.dfdx(x)

    def pred(self, l, x, w, b):
        raise NotImplementedError

    def grad_x(self, l, x, e, w, b, train):
        raise NotImplementedError

    def grad_w(self, l, x, e, w, b):
        raise NotImplementedError

    def grad_b(self, l, x, e, w, b):
        raise NotImplementedError


class PCN_AMB(PCNStructure):
    """
    PCGNtructure class with convention mu = wf(x)+b.
    """
    def pred(self, l, x, w, b):
        k = l - 1 if self.upward else l + 1
        bias = b[k] if self.use_bias else 0
        out = torch.matmul(self.fl(x[k], l), w[k])
        return out + bias

    def grad_x(self, l, x, e, w, b, train):
        k = l + 1 if self.upward else l - 1

        if l != self.L:
            grad = e[l] - self.dfldx(x[l], k) * (torch.matmul(e[k], w[l].T))
        else:
            if train:
                grad = 0
            else:
                if self.upward:
                    grad = e[l]
                else:
                    grad = -self.dfldx(x[l], k) * (torch.matmul(e[k], w[l].T))
        return grad

    def grad_w(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        return -torch.matmul(self.fl(x[l].T, k), e[k])

    def grad_b(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        return -e[k]


class PCN_MBA(PCNStructure):
    """
    PCNStructure class with convention mu = f(xw+b).
    """
    def pred(self, l, x, w, b):
        k = l - 1 if self.upward else l + 1
        bias = b[k] if self.use_bias else 0
        out = torch.matmul(x[k], w[k])
        return self.fl(out + bias, l)

    def grad_x(self, l, x, e, w, b, train):
        k = l + 1 if self.upward else l + 1
        bias = b[l] if self.use_bias else 0

        if l != self.L:
            temp = torch.matmul(x[l], w[l]) + bias
            grad = e[l] - torch.matmul(e[k] * self.dfldx(temp, k), w[l].T)
        else:
            if train:
                grad = 0
            else:
                if self.upward:
                    grad = e[l]
                else:
                    temp = torch.matmul(x[l], w[l]) + bias
                    grad = -torch.matmul(e[k] * self.dfldx(temp, k), w[l].T)
        return grad

    def grad_w(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        bias = b[l] if self.use_bias else 0
        temp = e[k] * self.dfldx(torch.matmul(x[l], w[l]) + bias, k)
        return -torch.matmul(x[l].T, temp)

    def grad_b(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        bias = b[l] if self.use_bias else 0
        return -e[k] * self.dfldx(torch.matmul(x[l], w[l]) + bias, k)  # same calc as grad_w so could make more efficient
