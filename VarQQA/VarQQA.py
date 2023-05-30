from operator import mod
import os
import pickle
import numpy as np
import torch
import scipy.optimize
import scipy.linalg
import collections
from tqdm import tqdm

from utils import unpackbits, exact_map, hamming_map
from utils import real_matrix_to_unitary, get_measure_matrix, hf_callback_wrapper

import torch_wrapper

np_rng = np.random.default_rng()
hf_data = lambda *x: os.path.join('data', *x)

class QueryQudit(torch.nn.Module):
    def __init__(self, num_bit, num_query, dim_query, partition, weight,
                 use_fractional=True, use_constraint=False, alpha_upper_bound=None, device='cpu',
                 sample_ratio=None, seed=None):
        super().__init__()
        partition = np.array(partition)
        dim_total = partition.sum()
        assert dim_total%dim_query==0
        self.weight = weight
        self.use_fractional = use_fractional
        if use_fractional:
            self.alpha = torch.nn.Parameter(torch.ones(num_query, dtype=torch.float64, device=device))
        else:
            self.alpha = torch.ones(num_query, dtype=torch.float64, device=device)
        np_rng = np.random.default_rng()
        tmp0 = np_rng.uniform(0, 2*np.pi, size=((num_query+1), dim_total, dim_total))
        self.theta = torch.nn.Parameter(torch.tensor(tmp0, dtype=torch.float64, device=device))
        self.measure_matrix = torch.tensor(get_measure_matrix(self.weight, partition).T, dtype=torch.float64, device=device)
        self.partition = partition
        self.num_bit = num_bit
        self.dim_total = dim_total
        self.dim_query = dim_query
        self.use_constraint = use_constraint
        self.sample_ratio = sample_ratio
        self.np_rng = np.random.default_rng(seed)
        tmp0 = np.arange(2**num_bit, dtype=np.uint64).reshape(-1,1)
        self.x_bit = torch.tensor(unpackbits(tmp0, axis=-1, count=dim_query, bitorder='l').T, dtype=torch.int64, device=device)
        self.state = None
        self.loss = None
        self.error_rate = None
        self.alpha_upper_bound = num_query if (alpha_upper_bound is None) else alpha_upper_bound
        self.device = device


    def unitary(self):
        return [real_matrix_to_unitary(x, with_phase=True).detach().cpu().numpy() for x in self.theta]
    
    def forward(self):
        XZ = [real_matrix_to_unitary(x, with_phase=True) for x in self.theta]
        alpha = self.alpha % 2
        oracle = [torch.exp((-1j*np.pi*x)*self.x_bit) for x in alpha]
        state = torch.zeros(self.dim_total, 2**self.num_bit, dtype=torch.complex128, device=self.device)
        state[0] = 1
        for ind0 in range(len(oracle)):
            state = XZ[ind0] @ state
            tmp0 = oracle[ind0].reshape(self.dim_query,1,-1)
            state = (tmp0 * state.reshape(self.dim_query, -1, state.shape[1])).reshape(self.dim_total, -1)
        state = XZ[ind0+1] @ state

        if self.sample_ratio is None:
            loss = -torch.sum(self.measure_matrix*torch.abs(state)**2)
        else:
            tmp0 = self.measure_matrix.shape[1]
            tmp1 = self.np_rng.choice(tmp0, size=int(tmp0*self.sample_ratio), replace=False, shuffle=False)
            loss = -torch.sum(self.measure_matrix*torch.abs(state)**2, dim=0)[torch.tensor(tmp1, device=self.device)].sum()
        self.error_rate = 1-torch.min(torch.sum(self.measure_matrix*torch.abs(state)**2,dim=0))
        self.state = state
        self.loss = loss
        if self.use_constraint:
            loss = loss + torch.square(torch.nn.functional.relu(alpha.sum()-self.alpha_upper_bound)).sum()
        return loss


def profile_model(model, row_limit=10):
    # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    is_cuda = next(model.parameters()).device.type=='cuda'
    tmp0 = [torch.profiler.ProfilerActivity.CPU] if is_cuda else [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=tmp0, record_shapes=True) as prof:
        with torch.profiler.record_function("model_inference"):
            model().backward()
    print(prof.key_averages().table(sort_by="cpu_time_total" if is_cuda else 'cuda_time_total', row_limit=row_limit))


def save_model(model,num_bit,num_modulo=None, k=None, l=None):
    num_query = len(model.alpha)
    num_bit = num_bit
    if num_modulo is None:
        tmp0 = '_frac' if model.use_fractional else ''
        filename = hf_data(f'exact/{num_bit}qubit_k{k}_l{l}_{num_query}query{tmp0}.pkl')
        with open(filename, "wb") as fid:
            tmp0 = {
                'alpha': model.alpha.cpu().detach().numpy().copy() if model.use_fractional else None,
                'num_qubit': num_bit,
                'num_query': num_query,
                'partition': model.partition,
                'theta': model.theta.cpu().detach().numpy().copy(),
                "error_rate":model.error_rate.item
            }
            pickle.dump(tmp0, fid)
    else:
        tmp0 = '_frac' if model.use_fractional else ''
        filename = hf_data(f'hamming/{num_bit}qubit_mod{num_modulo}_{num_query}query{tmp0}_part1.pkl')
        with open(filename, "wb") as fid:
            tmp0 = {
                'alpha': model.alpha.cpu().detach().numpy().copy() if model.use_fractional else None,
                "num_modulo": num_modulo,
                'num_qubit': num_bit,
                'num_query': num_query,
                'partition': model.partition,
                'theta': model.theta.cpu().detach().numpy().copy(),
                "error_rate":model.error_rate.item
            }
            pickle.dump(tmp0, fid)
    return filename


if __name__ == "__main__":
    num_bit = 5
    num_modulo = num_bit
    dim_query = num_bit +1
    num_query = 4
    use_fractional = False
    alpha_upper_bound = 0
    use_constrain = False
    function_type = "hamming"
    weight = hamming_map(num_bit, num_modulo)
    #function_type = "exact"
    #k=1,l=1
    #eight = exact_map(num_bit, k,l)
    print(collections.Counter(weight.tolist()))
    partition = [2,1,4,4,1]
    print(f"{num_bit=}\t{2**num_bit}\t{num_query=}\t{partition=}")
    
    model = QueryQudit(num_bit, num_query, dim_query, partition, weight,
            use_fractional=use_fractional, use_constraint=use_constrain, alpha_upper_bound=alpha_upper_bound, device='cuda')
    hf_model = torch_wrapper.hf_model_wrapper(model)
    loss_history = None
    hf_callback = hf_callback_wrapper(hf_model, model, loss_history,print_freq=100)
    num_parameter = torch_wrapper.get_model_flat_parameter(model).size
    theta0 = np_rng.uniform(-1, 1, size=num_parameter)
    theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B', jac=True, options={'maxiter':10000}, tol=1e-18, callback=hf_callback)
    print(theta_optim.fun)
    print(model.error_rate)
    tmp0 = dict(num_modulo=num_modulo, k=None, l=None) if (function_type=='hamming') else dict(num_modulo=None, k=k, l=l)
    save_model(model,num_bit,**tmp0)

    #alpha = model.alpha.detach().cpu().numpy()%2
    # #print(f'(n={num_bit},m={num_modulo},d={num_query}) dim_query={dim_query}, partition={partition}')
    #print(f'alpha: sum({alpha})={alpha.sum()}')
    # print('success_rate:', -model.loss/(2**num_bit))