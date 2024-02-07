
#### Object Oriented Desing For Implementation

- OOP is needed for simplifying the whole process of programming neural network's models

#### Components needed

##### 1 - Module
- Contains: 
1. Model
2. Losses functions
3. Optimization Methods

##### 2 -DataModule 
-  Contains DataLoaders for
1. Training
2. Validation

#### Utilities

- Utilities are needed to simplify OOP in jupyter notebooks

Utility that provide a functionality to add a function to a class outside of the class 
```python 
def add_to_class(Class):
	def wrapper(obj):
		setattr(Class, obj.__name__, obj)
		return wrapper
```
that can be used like this
```python
Class Lol:
	def __init__(self):
		self.a = 1337
lol  = Lol()

# Define a function for lol and add it with wrapper
@add_to_class(Lol)
def something(self):
	print(f"lol has attribute a setted in {self.a}")
	return self.a
```

##### Utility that saves all arguments in a class's __init__ method

```python
@d2l.add_to_class(d2l.HyperParameters)  #@save
def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k:v for k, v in local_vars.items()
                    if k not in set(ignore+['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
        setattr(self, k, v)
```

#### Models

- The Module class is the base class of all models, we always need to implement those methods:
1.  The **__init__** method that stores the learnable parameters
2. The **training_step** method gets a data batch and return the loss value
3. the **configure_optimizers** that returns the optimization method, or a list of optimization methods
4. Optionally, we can define a **validation_step** to report evaluaton of each epoch
5. Sometimes **forward** method

```python
class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

nn.module is the pytorch subclass that contains features for handling neural network.

##### Pytorch nn module:

For example, if we define a forward method, for example forward(self, X), then for an instance we can invoke this method by a(X). This work because it calls the **forward** method in **__call__**

#### DataModule 

-  A dataloader is a python generator that yields a data batch each time it is used. This batch is fed into the **training_step** method of Module to compute loss
1. Downloading the data
2. Preprocessing de data
- the **train_dataloader__** returns the dataloader for the training dataset. 
- This batch is fed into the **training_step** method defined in the module
```python
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```


#### Trainer

The **Trainer Class** is the class that trains the Module with the data defined in the DataModule, they key method is **fit** 
- **fit** method accepts two arguments, model and datamodel

```python
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

