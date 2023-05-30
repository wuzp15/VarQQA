import cvxpy as cvx
import numpy as np

def unpackbits(x,count):
    return [int(x&(1<<i)>0) for i in range(count)]

def hamming(x):
    return sum(unpackbits(x,int(np.log2(x+1)+1)))

def hamming_map(n,m):
    tmp = np.zeros(1<<n,dtype=np.int64)
    for i in range(1<<n):
        tmp[i] = hamming(i)%m
    return tmp

def exact_map(num_bit,ind_list):
    if type(ind_list)!= list:
        ind_list=[ind_list]
    ret = np.zeros(2**num_bit,dtype=np.int64)
    for i in range(1<<num_bit):
        ret[i] = (hamming(i) in ind_list)
    return ret

def majority_map(nbit):
    half = nbit//2
    assert half*2+1 == nbit
    maj_list = np.zeros(1<<nbit,dtype=np.int64)
    for i in range(1<<nbit):
        maj_list[i] = (hamming(i)>half)
    return maj_list

def OR_map(nbit):
    #half = nbit//2
    #assert half*2+1 == nqubit
    maj_list = np.zeros(1<<nbit,dtype=np.int64)
    for i in range(1<<nbit):
        maj_list[i] = (hamming(i)>0)
    return maj_list

def AND_map(nbit):
    #half = nbit//2
    #assert half*2+1 == nbit
    maj_list = np.zeros(1<<nbit,dtype=np.int64)
    for i in range(1<<nbit):
        maj_list[i] = (hamming(i)==nbit)
    return maj_list

def oracle_boolean(nbit,value_map):
    ## value_map is the map from bit string to output value
    m = max(value_map) + 1
    F_list=[]
    for i in range(m):
        F_list.append(np.eye(1<<nbit))
    for i in range(1<<nbit):
        for j in range(m):
            F_list[j][i][i] = (value_map[i] == j)  
    return F_list


def construct_E(nbit):
    E_list = [np.ones((1<<nbit,1<<nbit))]
    x_bit = [unpackbits(i,nbit) for i in range(1<<nbit)]
    for q in range(nbit):
        E = np.zeros((1<<nbit,1<<nbit))
        for i in range(1<<nbit):
            for j in range(1<<nbit):
                E[i][j] = (-1)**((x_bit[i][q]+x_bit[j][q]))
        E_list.append(E)
    return E_list


def sdp_boolean(nbit,boolean_map,t,verbose=False,max_iter=3000):
    '''
    params: 
        nbit: number of qubit or bit
        boolean_map: map the bit string to output value
        t : number of query
    '''
    E_list = construct_E(nbit)    
    F_list = oracle_boolean(nbit,boolean_map)
    M_list=[]
    G_list=[]
    constrains=[]
    m = max(boolean_map)+1 ## output domain start from 0
    for i in range(t):
        M_list.append([])
        for j in range(nbit+1):
            M_list[i]. append(cvx.Variable((1<<nbit,1<<nbit),symmetric=True))
    for i in range(m):
        G_list.append(cvx.Variable((1<<nbit,1<<nbit),symmetric=True))

    epss= cvx.Variable()
    constrains+=[epss>=0]
    ## semidefinite constrains
    for g in G_list:
        constrains+=[g>>0]
    for i in range(t):
        for j in range(nbit+1):
            constrains+=[M_list[i][j]>>0]

    ## input condition
    constrains+=[cvx.sum(M_list[0]) == np.ones((1<<nbit,1<<nbit))]

    ## running condition
    for i in range(1,t):
        tmp = np.zeros_like(E_list[0])
        for j in range(nbit+1):
            tmp = tmp+cvx.multiply(E_list[j],M_list[i-1][j])
        constrains+=[cvx.sum(M_list[i]) == tmp]

    ##ouput matches last but one query
    tmp = np.zeros_like(E_list[0])
    for i in range(nbit+1):
        tmp = tmp + cvx.multiply(E_list[i],M_list[t-1][i])
    constrains+=[cvx.sum(G_list) == tmp]

    ## measurement error rate condition
    for i in range(m):
        constrains+=[cvx.multiply(cvx.diag(G_list[i]),cvx.diag(F_list[i]))>=(1-epss)*cvx.diag(F_list[i])]
        #constrains+=[cvx.diag(G_list[i])>=(1-epss)*cvx.diag(F_list[i])]

    prob =cvx.Problem(cvx.Minimize(epss),constrains)
    prob.solve(solver=cvx.SCS, eps=1e-15,max_iters=max_iter,verbose=verbose)
    G_data = [g.value for g in G_list]
    M_data =[]
    for i in range(t):
        M_data.append([])
        for j in range(nbit+1):
            M_data[i].append(M_list[i][j].value)
    data = {"nbit":nbit,
            "modulo":m,
            "query":t,
            "M_list":M_data,
            "G_list":G_data,
            "epss":epss.value
    }
    return data


if __name__ == "__main__":
    nbit = 5
    boolean_map = hamming_map(nbit,nbit)
    nquery = 3
    print(f"{nbit=} \t {nquery=} \t {boolean_map=}")
    result = sdp_boolean(nbit,boolean_map,nquery,verbose=True,max_iter=2000)
    print(f"{1-result['epss']}")