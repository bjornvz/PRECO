import PRECO.optim as optim
from PRECO.structure import *

# NOTE: throughout in the comments I use [x0, x1, x2, x3] = [input, hidden 1, hidden 2, output]
# [e0, e1, e2, e3] = [0, error 1, error 2, output error]
# [w0, w1, w2]

class PCnet(PCmodel):        
    def __init__(self, lr_x: float, T_train: int, structure: PCNStructure, 
                 incremental: bool = False, min_delta: float = 0, early_stop: bool = False,
                 use_feedforward_init: bool = True, node_init_std: float = None, 
                 ):
        
        super().__init__(structure, lr_x, T_train, incremental, min_delta, early_stop)
        
        self.use_feedforward_init = use_feedforward_init
        self.node_init_std = node_init_std

        if use_feedforward_init:
            self.init_hidden = self.init_hidden_feedforward
            if node_init_std is not None:
                raise ValueError('Standard deviation should not be provided when using feedforward.')
        else:
            self.init_hidden = self.init_hidden_random
            if node_init_std is None:
                raise ValueError('Standard deviation must be provided when not using feedforward.')

        self.L = len(self.structure.layers)-1 # number of weight matrices

        if self.structure.upward:
            self.error_layers = range(1, self.L+1)
            self.weight_layers = range(0, self.L) 
        else:
            self.error_layers = range(0, self.L)
            self.weight_layers = range(1, self.L+1)
        self.hidden_layers = range(1, self.L) # x1, x2

        self._reset_grad()
        self._reset_params()

    @property
    def hparams(self):
        return {"lr_x": self.lr_x, "T_train": self.T_train, "incremental": self.incremental, "min_delta": self.min_delta,
                "early_stop": self.early_stop, "use_feedforward_init": self.use_feedforward_init, "node_init_std": self.node_init_std}
        
    @property
    def params(self):
        w = self.w[:-1] if self.structure.upward else self.w[1:]
        b = self.b[:-1] if self.structure.upward else self.b[1:]
        return {"w": w, "b": b, "use_bias": self.structure.use_bias}
    
    @property
    def grads(self):
        dw = self.dw[:-1] if self.structure.upward else self.dw[1:]
        db = self.db[:-1] if self.structure.upward else self.db[1:]
        return {"w": dw, "b": db}

    def _reset_grad(self):
        self.dw = [None for _ in range(self.L+1)]
        self.db = [None for _ in range(self.L+1)]


    def _reset_params(self):
        self.w, self.b = [], []
        for l in self.weight_layers:
            in_size = self.structure.layers[l] if self.structure.upward else self.structure.layers[l+1]
            out_size = self.structure.layers[l+1] if self.structure.upward else self.structure.layers[l]
            
            self.w.append(torch.empty( in_size, out_size, device=DEVICE))  
            self.weight_init(self.w[l])

            if self.structure.use_bias:
                self.b.append(torch.empty( out_size, device=DEVICE))
                self.bias_init(self.b[l])
       
        k = self.L if self.structure.upward else 0  # up convention: w0, w1, w2; down convention: w1, w2, w3
        self.w.insert(k, torch.empty(0,0, device=DEVICE)) 
        self.b.insert(k, torch.empty(0, device=DEVICE))
                
        self.no_weigths = sum([wl.shape[0]*wl.shape[1] for wl in self.w])
        if self.structure.use_bias:
            self.no_weigths += sum([bl.shape[0] for bl in self.b])

    def reset_nodes(self):
        self.e = [[] for _ in range(self.L+1)]
        self.x = [[] for _ in range(self.L+1)]


    def clamp_input(self, inp):
        self.x[0] = inp.clone()

    def clamp_target(self, target):
        self.x[-1] = target.clone()

    def init_hidden_random(self, batch_size):
        for l in self.hidden_layers:
            self.x[l] = torch.normal(0, self.node_init_std, size=(batch_size, self.structure.layers[l]), device=DEVICE)

    def init_hidden_feedforward(self, batch_size):
        self.forward(self.hidden_layers)

    def forward(self, layers):
        for l in layers:
            self.x[l] = self.structure.pred(l, self.x, self.w, self.b)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_updates(self, batch_no=None):
        for l in self.error_layers:
            self.e[l] = self.x[l] - self.structure.pred(l, self.x, self.w, self.b)
        
        if self.early_stop:
            early_stopper = optim.EarlyStopper(patience=0, min_delta=self.min_delta)

        for t in range(self.T_train): 
            for l in self.hidden_layers: 
                dEdx = self.structure.grad_x(l=l, train=True, x=self.x, e=self.e, w=self.w, b=self.b)
                self.x[l] -= self.lr_x*dEdx

            for l in self.error_layers:
                self.e[l] = self.x[l] - self.structure.pred(l, self.x, self.w, self.b)

            if self.incremental and self.dw.count(None) <= 1:  # incremental EM: update weights at every t
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])

            if self.early_stop:
                if early_stopper.early_stop( self.get_energy() ):
                    print(f"\nEarly stopping inference at t={t}.")          
                    break        

    def update_w(self):
        for l in self.weight_layers: # w0, w1, w2
            dEdw = self.structure.grad_w(l, self.x, self.e, self.w, self.b)
            if self.structure.use_bias:
                dEdb = self.structure.grad_b(l, self.x, self.e, self.w, self.b)

            self.dw[l] = dEdw
            if self.structure.use_bias:
                self.db[l] = torch.sum(dEdb, axis=0)

    def train_supervised(self, X_batch, y_batch, batch_no=None):
        X_batch = to_vector(X_batch)      # makes e.g. 28*28 -> 784
        y_batch = onehot(y_batch, N=self.structure.layers[-1])   

        self.reset_nodes()
        self.clamp_input(X_batch)
        self.init_hidden(X_batch.shape[0])
        self.clamp_target(y_batch)
        self.train_updates(batch_no=batch_no)
        self.update_w()
        if not self.incremental:
            self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])

    
    def test_supervised(self, X_batch):
        X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        self.reset_nodes()
        self.clamp_input(X_batch)
        self.forward(self.error_layers)
        return self.x[self.L]

    def get_errors(self):
        temp = []
        for l in self.error_layers:
            temp.append(torch.mean(self.e[l], axis=0))
        return torch.concatenate(temp)

    def get_energy(self):
        return torch.sum( self.get_errors()**2 ).item()

    def get_weights(self):
        w = []
        for l in self.weight_layers:
            w.append(self.w[l].clone())
        return w
    
    def get_mean_weights(self):
        w = []
        for l in self.weight_layers:
            w.append(torch.mean(self.w[l]))
        return w