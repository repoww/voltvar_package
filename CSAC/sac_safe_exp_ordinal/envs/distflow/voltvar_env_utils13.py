import numpy as np

def obtain_network_data(vref=1.0,load=1.0):
    # function: obtain_network_data: a simple data storage function, the format of the data is similar to matpower data
    # IEEE 16 bus data:
    # bus format:   bus num, type, Sd (in p.u.), Sg (in p.u.)
    # note that, Sd is going to be substituted by modified CALLI VALLI data. The values showing here are just
    # sample values (the original IEEE 16 bus values). Note: place all reference buses at the top of the list
    # input: Feeder_size: an integer representing the number of node in the feeder
    bus = np.array([[1, 1, 0],
                    [2, 2, 0.015333+0.0088j],
                    [3, 2, 0.011333+0.0083333j],
                    [4, 2, 0],
                    [5, 2, 0],
                    #[6, 2, 0.1+0.04j],
                    [6, 2, 0.04 + 0.029j],
                    [7, 2, 0.0088889+0.0051556j],
                    [8, 2, 0.011333+0.0053333j],
                    [9, 2, 0],
                    [10, 2, 0.092778+0.056578j],
                    [11, 2, 0.0562+0.0308j],
                    [12, 2, 0.0085333+0.0057333j],
                    [13,2,0]
                    ])
    bus[:,2] *= load
    # line format: from bus, to bus, line impedance z = r+jx (in p.u.)
    line = np.array([[1,4,0.061166,0.19593],
                     [2,3,0.055158,0.043974],
                     [3,4,0.09193,0.07329],
                     [4,5,0.048597,0.0624],
                     [5,6,0.33,0.6],
                     [4,7,0.015292,0.048983],
                     [7,10,0.045875,0.14695],
                     [8,9,0.065461,0.066362],
                     [9,10,0.055158,0.043974],
                     [10,11,0.040006,0.034074],
                     [9,12,0.17631,0.067293],
                     [10,13,0.030583,0.097966]
                    ])
    # line = np.array([[1, 2, 3.34403, 7.88777],
    #                  [2, 3, 5.00, 300.0],
    #                  [3, 4, 37.560096, 88.482919]
    #                  ])
    v2ref = vref**2  # reference node voltage
    return bus, line, v2ref



def load_flow(action,load):
    reg_tap,tsf_tap,cap_tap = action
    # vref = 0.95+reg_tap*0.01
    # tsf_pos = tsf_tap*0.01+0.95 # from node 5 to node 6 line[4]
    # cap_pos = 0.04*cap_tap   # 200KVA, base 5MVA, at node 11/node[10]
    vref = 1 + reg_tap * 0.05
    tsf_pos = tsf_tap * 0.05 + 1
    cap_pos = 0.01 + 0.01 * cap_tap
    bus, line, v2ref = obtain_network_data(vref,load)
    num_edge = line.shape[0]
    num_node = bus.shape[0]
    num_ref_node = 1
    num_nonref_node = num_node - num_ref_node

    P_spec = -np.real(bus[1:,2])
    Q_spec = -np.imag(bus[1:,2])

    App = np.zeros((num_nonref_node, num_edge))
    Apq = np.zeros((num_nonref_node, num_edge))
    Apv = np.zeros((num_nonref_node, num_node))
    Apl = np.zeros((num_nonref_node, num_edge))
    Aqp = np.zeros((num_nonref_node, num_edge))
    Aqq = np.zeros((num_nonref_node, num_edge))
    Aqv = np.zeros((num_nonref_node, num_node))
    Aql = np.zeros((num_nonref_node, num_edge))
    Avp = np.zeros((num_ref_node, num_edge))
    Avq = np.zeros((num_ref_node, num_edge))
    Avv = np.zeros((num_ref_node, num_node))
    Avl = np.zeros((num_ref_node, num_edge))
    Alp = np.zeros((num_nonref_node, num_edge))
    Alq = np.zeros((num_nonref_node, num_edge))
    Alv = np.zeros((num_nonref_node, num_node))
    All = np.zeros((num_nonref_node, num_edge))

    for ii in range(num_ref_node, num_node):  # looping over non-reference nodes
        i_set = np.where(line[:, 0] == bus[ii, 0])
        j_set = np.where(line[:, 1] == bus[ii, 0])

        # first linear eqn
        if i_set[0].size != 0:
            App[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            App[ii - num_ref_node, j_set] = -1
            Apl[ii - num_ref_node, j_set] = line[j_set, 2]

        # second linear eqn
        if i_set[0].size != 0:
            Aqq[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            Aqq[ii - num_ref_node, j_set] = -1
            Aql[ii - num_ref_node, j_set] = line[j_set, 3]

        if ii==10:
            Aqv[ii - num_ref_node,ii] = -cap_pos

    # third linear eqn
    Avv[:num_ref_node, :num_ref_node] = np.identity(num_ref_node)

    # forth linear eqn
    for ij in range(num_edge):
        Alp[ij, ij] = -2 * line[ij, 2]
        Alq[ij, ij] = -2 * line[ij, 3]
        Alv[ij, int(line[ij, 0] - 1)] = 1  # from node， the minus 1 is for matching python indexing convention
        if ij==4:
            Alv[ij, int(line[ij, 1] - 1)] = -1/tsf_pos**2 # to node
        else:
            Alv[ij, int(line[ij, 1] - 1)] = -1  # to node
        All[ij, ij] = line[ij, 2] ** 2 + line[ij, 3] ** 2

    A = np.vstack((np.hstack((App, Apq, Apv, Apl)),
                   np.hstack((Aqp, Aqq, Aqv, Aql)),
                   np.hstack((Avp, Avq, Avv, Avl)),
                   np.hstack((Alp, Alq, Alv, All))))
    b = np.vstack((P_spec.reshape((len(P_spec), 1)),
                   Q_spec.reshape((len(Q_spec), 1)),
                   np.ones((num_ref_node, 1))*v2ref,
                   np.zeros((num_edge, 1))))

    # initial guess
    Pij_ini = np.zeros((num_edge, 1))
    Qij_ini = np.zeros((num_edge, 1))
    Vi2_ini = np.ones((num_node, 1))
    Lij_ini = np.zeros((num_edge, 1))
    x = np.vstack((Pij_ini, Qij_ini, Vi2_ini, Lij_ini))

    # being N-R procedure
    dx = [1000, 1]
    iter = 0
    convergence_flag = 1 # if convergence_flag == 1, then it is converged
    while max(np.absolute(dx)) > 0.00001:
        # mismatch vector of the quadratic equation
        Dg = x[:num_edge] ** 2 + x[num_edge:2 * num_edge] ** 2 - (
            np.multiply(x[2 * num_edge + num_node:], x[2 * num_edge + (line[:, 0] - 1).astype(
                int)]))  # numpyArray.astype(int): convert to integer element-wise
        # mismatch jacobian of the quadratic equation
        Gp = np.zeros((num_edge, num_edge))
        Gq = np.zeros((num_edge, num_edge))
        Gv = np.zeros((num_edge, num_node))
        Gl = np.zeros((num_edge, num_edge))
        for ij in range(num_edge):
            Gp[ij, ij] = 2 * x[ij]  # 2*pij
            Gq[ij, ij] = 2 * x[ij + num_edge]
            Gv[ij, int(line[ij, 0] - 1)] = -x[2 * num_edge + num_node + ij] # the minus 1 is for matching python indexing convention
            Gl[ij, ij] = -x[2 * num_edge + int(line[ij, 0] - 1)] # the minus 1 is for matching python indexing convention
        DG = np.hstack((Gp, Gq, Gv, Gl))

        # overall mismatch vector
        Df = np.vstack((A.dot(x) - b, Dg))  # where A.dot(x) is matrix-vector multi for numpy array A and x
        # overall mismatch Jacobian
        J = np.vstack((A, DG))

        # solve
        dx = np.linalg.solve(-J, Df)

        # update
        x = x + dx

        iter += 1
        if iter > 20:
            # print("does not converge")
            convergence_flag = 0
            break
    Pij = x[:num_edge]
    Qij = x[num_edge:2 * num_edge]
    Vi2 = x[2 * num_edge:2 * num_edge + num_node]
    Lij = x[2 * num_edge + num_node:]
    if convergence_flag == 1:  # if the load flow converges
        total_loss = line[:, 2].dot(Lij)
    else:  # if the load flow does not converge
        total_loss = 10  # assign to total_loss a very big number (converged program should have total_loss < 1)
    return Pij, Qij, Vi2, Lij, total_loss[0], iter, convergence_flag





