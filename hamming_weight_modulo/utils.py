import numpy as np

def hamming_map(num_bitstring, num_modulo):
    ret = np.zeros(2**num_bitstring, dtype=np.int64)
    for i in range(2**num_bitstring):
        ret[i] = sum(x == "1" for x in bin(i)[2:]) % num_modulo
    return ret

def hamming(x):
    return sum( [x&(1<<i)>0 for i in range(16)] )


def to_binary(x,bit_len):
    return np.array([x&(1<<i)>0 for i in range(bit_len)])


def unpackbits(x,count,axis=-1,bitorder='l'):
    tmp0 = x.view(np.uint64).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(tmp0,count=count,axis=axis, bitorder=bitorder)


def exact_map(num_bit,ind_list):
    if type(ind_list)!= list:
        ind_list=[ind_list]
    ret = np.zeros(2**num_bit,dtype=np.int64)
    for i in range(1<<num_bit):
        ret[i] = (hamming(i) in ind_list)
    return ret

def mes_matrix(partition,weight):
    ## partition is the partition of accessible space.
    # If dim_acc=5.partition can be[1,2,2]
    ## weight is a list of hamming modulo. can be computed by calling hamming_map(n,m)
    dim_bit =len(weight)
    ind = np.cumsum([0,*partition])
    mes = np.zeros((dim_bit,ind[-1]),dtype=np.complex128)
    for i in range(dim_bit):
        mes[i][ind[weight[i]]:ind[weight[i]+1]]=1
    return mes.T


def oracle(num_bit,dim_query,dim_work):
    # construct the orcale for all x
    # num_bit : number of bit for boolean function
    # dim_query : dimension of query register
    # dim_work : dimension of working memory register
    tmp0 = np.arange(2**num_bit, dtype=np.uint64).reshape(-1,1)
    x_bit = unpackbits(tmp0, axis=-1, count=dim_query, bitorder='l').T
    dim_acc = dim_query * dim_work
    oracle = np.zeros([dim_acc,2**num_bit],dtype=np.complex128)
    for i in range(2**num_bit):
        oracle[:,i] = np.kron((-1)**x_bit[:,i],np.ones(dim_work))
    return oracle

def forward(dim_acc,num_bit,unitary,oracle):
    state = np.zeros([dim_acc,2**num_bit],dtype=np.complex128)
    state[0] = 1
    state = unitary[0]@state
    for i in range(1,len(unitary)):
        state = oracle*state
        state = unitary[i]@state
    return state

def record_state(dim_acc,num_bit,unitary,oracle):
    state_list=[]
    state = np.zeros([dim_acc,2**num_bit],dtype=np.complex128)
    state[0] = 1
    state = unitary[0]@state
    for i in range(1,len(unitary)):
        state = oracle*state
        state_list.append(state)
        state = unitary[i]@state
    return state_list