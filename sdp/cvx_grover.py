import numpy as np
import cvxpy as cvx


def construct_Grover_E(nqubit):
    E_list = [np.ones((1<<nqubit,1<<nqubit))]
    for q in range(1<<nqubit):
        E = np.zeros((1<<nqubit,1<<nqubit))
        for i in range(1<<nqubit):
            for j in range(1<<nqubit):
                a = (i==q)
                b = (j==q)
                E[i][j] = (-1)**((a+b)%2)
        E_list.append(E)
    return E_list

def oracle_Grover(nqubit):
    F_list = []
    for i in range(1<<nqubit):
        tmp=np.zeros((1<<nqubit,1<<nqubit))
        tmp[i][i]=1
        F_list.append(tmp)
    return F_list


def sdp_grover(nqubit,t,verbose=False,max_iter=3000):
    '''
    params: 
        nqubit: number of qubit
        t : number of query
    '''
    E_list = construct_Grover_E(nqubit)    
    F_list = oracle_Grover(nqubit)
    M_list = []
    G_list = []
    epss = cvx.Variable()
    constrains = []
    #constrains+=[epss>=0]

    for i in range(t):
        M_list.append([])
        for j in range((1<<nqubit)):
            M_list[i].append(cvx.Variable((1<<nqubit,1<<nqubit),symmetric=True))
    for i in range(1<<nqubit):
        G_list.append(cvx.Variable((1<<nqubit,1<<nqubit),symmetric=True))

    ## semidefinite constrains
    for g in G_list:
        constrains += [g>>0]
    for i in range(t):
        for j in range((1<<nqubit)):
            constrains += [M_list[i][j]>>0]

    ## input condition
    constrains += [cvx.sum(M_list[0]) == np.ones((1<<nqubit,1<<nqubit))]

    ## running condition
    for i in range(1,t):
        tmp = np.zeros_like(E_list[0])
        for j in range((1<<nqubit)):
            tmp = tmp+cvx.multiply(E_list[j+1],M_list[i-1][j])
        constrains += [tmp == cvx.sum(M_list[i])]

    ##ouput matches last but one query
    tmp = np.zeros_like(E_list[0])
    for i in range((1<<nqubit)):
        tmp = tmp + cvx.multiply(E_list[i+1],M_list[t-1][i])
    constrains+=[cvx.sum(G_list) == tmp]


    for i in range(1<<nqubit):
        constrains+=[cvx.multiply(cvx.diag(G_list[i]),cvx.diag(F_list[i]))>=(1-epss)*cvx.diag(F_list[i])]

    prob =cvx.Problem(cvx.Minimize(epss),constrains)
    prob.solve(solver=cvx.SCS, eps=1e-15,max_iters=max_iter,verbose=verbose)

    G_data = [g.value for g in G_list]
    M_data =[]
    for i in range(t):
        M_data.append([])
        for j in range(nqubit+1):
            M_data[i].append(M_list[i][j].value)
    data = {"nqubit":nqubit,
            "query":t,
            "M_list":M_data,
            "G_list":G_data,
            "epss":epss.value
    }
    return data


if __name__ == "__main__":
    ## For grover search problem, Mosek solver perform better than SCS solver.
    nqubit = 4
    nquery = 2
    result = sdp_grover(nqubit, nquery, verbose=True, max_iter=4000)
    print(f"success rate:{1-result['epss']}")