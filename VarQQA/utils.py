import itertools
import numpy as np
import scipy.linalg
import torch
import time


def hamming_map(num_bitstring, num_modulo):
    ret = np.zeros(2**num_bitstring, dtype=np.int64)
    for i in range(2**num_bitstring):
        ret[i] = sum(x == "1" for x in bin(i)[2:]) % num_modulo
    return ret

def hamming(x, bit_len=16):
    assert x < 2**bit_len
    return sum( [x&(1<<i)>0 for i in range(bit_len)] )


def to_binary(x,bit_len):
    return np.array([x&(1<<i)>0 for i in range(bit_len)])


def unpackbits(x,count,axis=-1,bitorder='l'):
    tmp0 = x.view(np.uint64).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(tmp0,count=count,axis=axis, bitorder=bitorder)

def NE_function(bit):
    if len(bit)==3:
        return ((bit[0]==bit[1] and bit[1]==bit[2])+1)%2
    else:
        stride = len(bit)//3
        return NE_function([NE_function(bit[:stride]),NE_function(bit[stride:2*stride]),NE_function(bit[2*stride:])])

def test_function():
    x=np.arange(1<<8)
    x_bit = unpackbits(x,8)
    x1 = x_bit[:,:4]
    x2 = x_bit[:,4:]
    tmp1 = np.sum(x1,axis=1)%2
    tmp2 = np.sum(x2,axis=1)%2
    output = 2*tmp1+tmp2
    return output

def NE_map(num_bit):
    assert 3**(int(np.log(num_bit)/np.log(3)))==num_bit
    ret = np.zeros(2**num_bit,dtype=np.int64)
    for i in range(1<<num_bit):
        ret[i] = NE_function(to_binary(i,num_bit))
    return ret


def sort_hamming(n,m):
    ham= hamming_map(n,m)
    string_list =[i for i in range(1<<n)]
    string_list.sort(key= lambda x:ham[x])
    return string_list

def majority_map(num_bit):
    half = num_bit//2
    assert half*2 != num_bit
    majority_list = np.zeros(1<<num_bit,dtype=np.int64)
    for i in range(1<<num_bit):
        majority_list[i] = (hamming(i)>half)
    return majority_list

def exact_map(num_bit,ind_list):
    if type(ind_list)!= list:
        ind_list=[ind_list]
    ret = np.zeros(2**num_bit,dtype=np.int64)
    for i in range(1<<num_bit):
        ret[i] = (hamming(i) in ind_list)
    return ret

def compute_gram_matrix(ret,string_list):
    _,dim1 = ret.shape
    gram = 0.5*np.eye(dim1)
    for i in range(dim1):
        for j in range(i+1,dim1):
            gram[i,j] = torch.abs(torch.dot(ret[:,string_list[i]].conj(),ret[:,string_list[j]]))
    return gram+gram.T


def mes_matrix(partition,weight):
    dim_bit =len(weight)
    ind = np.cumsum([0,*partition])
    mes = torch.zeros(dim_bit,ind[-1],dtype=torch.complex128)
    for i in range(dim_bit):
        mes[i][ind[weight[i]]:ind[weight[i]+1]]=1
    return mes.T

def get_numpy_rng(np_rng_or_seed_or_none):
    if np_rng_or_seed_or_none is None:
        ret = np.random.default_rng()
    elif isinstance(np_rng_or_seed_or_none, np.random.Generator):
        ret = np_rng_or_seed_or_none
    else:
        seed = int(np_rng_or_seed_or_none)
        ret = np.random.default_rng(seed)
    return ret


def np_random_complex(*size, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=size + (2,)).astype(np.float64, copy=False).view(np.complex128).reshape(size)
    return ret


def real_matrix_to_unitary(matA, with_phase=False):
    shape = matA.shape
    matA = matA.reshape(-1, shape[-1], shape[-1])
    is_torch = isinstance(matA, torch.Tensor)
    if is_torch:
        tmp0 = torch.tril(matA, -1)
        tmp1 = torch.triu(matA)
        if with_phase:
            tmp3 = tmp1
        else:
            tmp2 = torch.diagonal(tmp1, dim1=-2, dim2=-1).sum(dim=1).reshape(-1,1,1)/shape[-1]
            tmp3 = tmp1 - tmp2*torch.eye(shape[-1], device=matA.device)
        tmp4 = 1j*(tmp0 - tmp0.transpose(1,2)) + (tmp3 + tmp3.transpose(1,2))
        ret = torch.linalg.matrix_exp(1j*tmp4)
    else:
        tmp0 = np.tril(matA, -1)
        tmp1 = np.triu(matA)
        if not with_phase:
            tmp1 = tmp1 - np.trace(tmp1, axis1=-2, axis2=-1).reshape(-1,1,1)/shape[-1]*np.eye(shape[-1])
        tmp2 = 1j*(tmp0 - tmp0.transpose(0,2,1)) + (tmp1 + tmp1.transpose(0,2,1))
        ret = np.stack([scipy.linalg.expm(1j*x) for x in tmp2])
        # ret = scipy.linalg.expm(1j*tmp2) #TODO scipy-v1.9
    ret = ret.reshape(*shape)
    return ret


def get_measure_matrix(hamming_weight, partition):
    partition = np.array(partition)
    ret = np.zeros((len(hamming_weight), partition.sum()), dtype=np.int64)
    tmp0 = np.cumsum(np.pad(partition, [(1,0)], mode='constant'))
    for ind0 in range(len(partition)):
        ret[hamming_weight==ind0,tmp0[ind0]:tmp0[ind0+1]] = 1
    return ret

def hf_callback_wrapper(hf_fval, model, state=None, print_freq=1):
    if state is None:
        state = dict()
    state['step'] = 0
    state['time'] = time.time()
    state['fval'] = []
    state['time_history'] = []
    state['error_rate'] =[]
    def hf0(theta):
        step = state['step']
        if (print_freq>0) and (step%print_freq==0):
            t0 = state['time']
            t1 = time.time()
            fval = hf_fval(theta, tag_grad=False)[0]
            print(f'[step={step}][time={t1-t0:.3f} seconds] loss={fval}, error_rate={model.error_rate.item()}')
            state['fval'].append(fval)
            state["error_rate"].append(model.error_rate.item())
            state['time'] = t1
            state['time_history'].append(t1-t0)
        state['step'] += 1
    return hf0