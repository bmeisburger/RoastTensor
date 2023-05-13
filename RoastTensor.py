import torch

class RoastTensor(object):
    def __init__(self, data, compression=1.0, requires_grad=False, seed=2023, **kwargs):
        
        t = torch.as_tensor(data, **kwargs)

        self.device = t.device
        self.requires_grad = t.requires_grad or requires_grad
        self.dtype = t.dtype
        self._orig_size = t.size()
        self._compression = compression
        self._comp_size = torch.Size([int(t.numel() * self._compression)])

        # See: https://en.wikipedia.org/wiki/Universal_hashing
        gen = torch.Generator()
        gen.manual_seed(seed)
        self._P = 1008863  # A large prime
        if (self._P < self._comp_size.numel()):
            raise ValueError(f"hashing prime ({self._P}) must be larger than compressed size ({self._comp_size})")
        
        self._a = torch.randint(low=1, high=self._P, size=(1,), generator=gen).item()
        self._b = torch.randint(low=0, high=self._P, size=(1,), generator=gen).item()
        self._hash = lambda x, m: ((self._a * x + self._b) % self._P) % m  # Universal hash function

        # idx = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int64, device=t.device).reshape(self._orig_size), self._comp_size.numel())
        # g = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int32, device=t.device).reshape(self._orig_size), 2) * 2 - 1
        
        # # Compress tensor
        # self._t = torch.zeros(self._comp_size, dtype=t.dtype, device=t.device, requires_grad=t.requires_grad)
        # self._t.scatter_add_(0, idx.view(-1), torch.mul(t, g).view(-1))

        # count = torch.zeros(self._comp_size, dtype=torch.float, device=t.device)
        # count.scatter_add_(0, idx.view(-1), torch.ones_like(idx, device=t.device, dtype=torch.float).view(-1)) + 1e-3

        # self._t = torch.div(self._t, count)

        self._t = self.compress(t)

        # compress gradient, if necessary
        if t.grad:
            self._t.grad = self.compress(t.grad)


        # print(self._t)
        # print(torch.mul(self._t[idx], g))

    
    def decompress(self, value: torch.Tensor):
        idx = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int64, device=self._t.device).reshape(self._orig_size), self._comp_size.numel())
        g = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int32, device=self._t.device).reshape(self._orig_size), 2) * 2 - 1
        
        return torch.mul(value[idx], g)


    def compress(self, value: torch.Tensor):
        # add check for device and dtype

        idx = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int64, device=self.device).reshape(self._orig_size), self._comp_size.numel())
        g = self._hash(torch.arange(start=0, end=self._orig_size.numel(), dtype=torch.int32, device=self.device).reshape(self._orig_size), 2) * 2 - 1

        t = torch.zeros(self._comp_size, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        t.scatter_add_(0, idx.view(-1), torch.mul(value, g).view(-1))

        count = torch.zeros(self._comp_size, dtype=torch.float, device=self.device)
        count.scatter_add_(0, idx.view(-1), torch.ones_like(idx, device=self.device, dtype=torch.float).view(-1)) + 1e-3

        return torch.div(t, count)


    def __add__(self, other):
        other = other.decompress(other._t) if hasattr(other, '_t') else other
        return RoastTensor(self.decompress(self._t) + other, compression=self._compression)


    @property
    def grad(self):
        return self.decompress(self._t.grad) if self._t.requires_grad else None
    
    @grad.setter
    def grad(self, value):
        self._t.grad = self.compress(value)

    @property
    def data(self):
        return self.decompress(self._t)
    
    @data.setter
    def data(self, value):
        self._t = self.compress(value)

    @property
    def dtype(self):
        return self._t.dtype
    
    @property
    def shape(self):
        return self._orig_size
    
    def size(self):
        return self._orig_size

    def __repr__(self):
        return "Compression:\n{}\n\ndata:\n{}".format(self._compression, self.decompress(self._t))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        compressions = tuple(a._compression for a in args if hasattr(a, '_compression'))
        args = [a.decompress(a._t) if hasattr(a, '_t') else a for a in args]
        assert len(compressions) > 0
        ret = func(*args, **kwargs)
        return RoastTensor(ret, compression=compressions[0])
