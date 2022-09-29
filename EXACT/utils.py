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