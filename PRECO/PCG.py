from PRECO.optim import *
from PRECO.structure import *

class PCgraph(PCmodel):
    """
    Predictive Coding Graph trained with Inference Learning (IL)/Expectation Maximization (EM).

    Args:
        lr_x (float): Inference rate/learning rate for nodes (partial E-step).
        T_train (int): Number of inference iterations (partial E-steps) during training.
        T_test (int): Number of inference iterations (partial E-steps) during testing.
        incremental (bool): Whether to use incremental EM (partial M-step after each partial E-step).
        init (dict): Initialization parameters: {"weights": std, "bias": std, "x_hidden": std, "x_output": std}.
        min_delta (float): Minimum change in energy for early stopping during inference.
        use_input_error (bool): Whether to use input error in training (to obtain exact PCN updates for hierarchical mask, set to False).
    """
    def __init__(self, lr_x: float, T_train: int, T_test: int, structure: PCGStructure, 
                 incremental: bool = False, 
                 node_init_std: float = None,
                 min_delta: float = 0, early_stop: bool = False,
                 use_input_error: bool = True):
         
        super().__init__(structure=structure, lr_x=lr_x, T_train=T_train, incremental=incremental, min_delta=min_delta, early_stop=early_stop)
        self.T_test = T_test

        self.use_input_error = use_input_error
        self.node_init_std = node_init_std

        self._reset_grad()
        self._reset_params()

        if self.structure.mask is not None:
            self.w = self.structure.mask * self.w 
            self.mask_density = torch.count_nonzero(self.w)/self.structure.N**2   
            # hierarchical structure
            if self.structure.num_layers is not None:
                self.test_supervised = self.test_feedforward
                self.init_hidden = self.init_hidden_feedforward
                if use_input_error:
                    logging.warning("Using input error in training with hierarchical mask (no input error recommended).")
            # non-hierarchical structure
            else: 
                self.test_supervised = self.test_iterative
                self.init_hidden = self.init_hidden_random
                if not use_input_error:
                    logging.warning("Not using input error in training with non-hierarchical mask.")


    @property
    def hparams(self):
        return {"lr_x": self.lr_x, "T_train": self.T_train, "T_test": self.T_test, "incremental": self.incremental,
                 "min_delta": self.min_delta,"early_stop": self.early_stop, "use_input_error": self.use_input_error, "node_init_std": self.node_init_std}

    @property
    def params(self):
        return {"w": self.w, "b": self.b, "use_bias": self.structure.use_bias}
    
    @property
    def grads(self):
        return {"w": self.dw, "b": self.db}

    def _reset_params(self):
        self.w = torch.empty( self.structure.N, self.structure.N, device=DEVICE)
        self.b = torch.empty( self.structure.N, device=DEVICE)
        
        self.weight_init(self.w)
        if self.structure.use_bias:
            self.bias_init(self.b)

    def _reset_grad(self):
        self.dw, self.db = None, None

    def reset_nodes(self, batch_size=1):
        self.e = torch.empty(batch_size, sum(self.structure.shape), device=DEVICE)
        self.x = torch.zeros(batch_size, sum(self.structure.shape), device=DEVICE)

    def clamp_input(self, inp):
        di = self.structure.shape[0]
        self.x[:,:di] = inp.clone()

    def clamp_target(self, target):
        do = self.structure.shape[2]
        self.x[:,-do:] = target.clone()
        
    def init_hidden_random(self):
        di = self.structure.shape[0]
        do = self.structure.shape[2]
        self.x[:,di:-do] = torch.normal(0.5, self.node_init_std,size=(self.structure.shape[1],), device=DEVICE)

    def init_hidden_feedforward(self):
        self.forward(self.structure.num_layers-1)

    def init_output(self):
        do = self.structure.shape[2]
        self.x[:,-do:] = torch.normal(0.5, self.node_init_std, size=(do,), device=DEVICE)

    def forward(self, no_layers):
        temp = self.x.clone()
        for l in range(no_layers):
            lower = sum(self.structure.layers[:l+1])
            upper = sum(self.structure.layers[:l+2])
            temp[:,lower:upper] = self.structure.pred(x=temp, w=self.w, b=self.b )[:,lower:upper]
        self.x = temp

    def update_w(self):
        self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
        if self.structure.use_bias:
            self.db = self.structure.grad_b(x=self.x, e=self.e, w=self.w, b=self.b)
            
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_xs(self, train=True):
        if self.early_stop:
            early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        di = self.structure.shape[0]
        upper = -self.structure.shape[2] if train else self.structure.N

        T = self.T_train if train else self.T_test
        
        for t in range(T): 
            self.e = self.x - self.structure.pred( x=self.x, w=self.w, b=self.b )
            if not self.use_input_error:
                self.e[:,:di] = 0 
            
            dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes
            self.x[:,di:upper] -= self.lr_x*dEdx 

            if self.incremental and self.dw is not None:
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])
                
            if self.early_stop:
                if early_stopper.early_stop( self.get_energy() ):
                    break            

    def train_supervised(self, X_batch, y_batch): 
        X_batch = to_vector(X_batch)                  # makes e.g. 28*28 -> 784
        y_batch = onehot(y_batch, N=self.structure.shape[2])    # makes e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]

        self.reset_nodes(batch_size=X_batch.shape[0])        
        self.clamp_input(X_batch)
        self.init_hidden()
        self.clamp_target(y_batch)

        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])


    def test_feedforward(self, X_batch):
        X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        self.reset_nodes(batch_size=X_batch.shape[0])
        self.clamp_input(X_batch)
        self.forward(self.structure.num_layers)

        return self.x[:,-self.structure.shape[2]:] 

    def test_iterative(self, X_batch, diagnostics=None, early_stop=False):
        X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        self.reset_nodes(batch_size=X_batch.shape[0])
        self.clamp_input(X_batch)
        self.init_hidden()
        self.init_output()

        self.update_xs(train=False, diagnostics=diagnostics, early_stop=early_stop)
        return self.x[:,-self.structure.shape[2]:] 

    def get_weights(self):
        return self.w.clone()

    def get_energy(self):
        return torch.sum(self.e**2).item()

    def get_errors(self):
        return self.e.clone()    