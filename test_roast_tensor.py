import torch
from RoastTensor import RoastTensor

def main():
    m = RoastTensor([[0., 2.], [-6., 4.]], compression=0.5, requires_grad=False)
    n = RoastTensor([[5., 5.], [5., 5.]], compression=0.5, requires_grad=True)
    # print(m.decompress(m._t))
    print(n.decompress(n._t))
    x = m + n
    print(x)
    print(x.grad)
    # func_dict = torch.overrides.get_overridable_functions()
    # tensor_funcs = func_dict[torch.Tensor]


def helper(t: torch.Tensor) -> torch.Tensor:
    t2 = torch.tensor([[1., 2.], [1., 2.]], requires_grad=False)
    return torch.mul(t, t2)

if __name__ == '__main__':
    main()