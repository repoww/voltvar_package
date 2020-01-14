import numpy as np

def obtain_network_data(vref=1.0,load=1.0):
    # function: obtain_network_data: a simple data storage function, the format of the data is similar to matpower data
    # IEEE 16 bus data:
    # bus format:   bus num, type, Sd (in p.u.), Sg (in p.u.)
    # note that, Sd is going to be substituted by modified CALLI VALLI data. The values showing here are just
    # sample values (the original IEEE 16 bus values). Note: place all reference buses at the top of the list
    # input: Feeder_size: an integer representing the number of node in the feeder
    bus = np.array([[1, 1, 0.000000 + 0.000000j],
                    [2, 2, 0.000000 + 0.000000j],
                    [3, 2, 0.001833 + 0.000967j],
                    [4, 2, 0.000000 + 0.000000j],
                    [5, 2, 0.000533 + 0.000267j],
                    [6, 2, 0.000000 + 0.000000j],
                    [7, 2, 0.000000 + 0.000000j],
                    [8, 2, 0.000000 + 0.000000j],
                    [9, 2, 0.000000 + 0.000000j],
                    [10, 2, 0.000000 + 0.000000j],
                    [11, 2, 0.001133 + 0.000567j],
                    [12, 2, 0.004500 + 0.002333j],
                    [13, 2, 0.000167 + 0.000067j],
                    [14, 2, 0.001333 + 0.000667j],
                    [15, 2, 0.000133 + 0.000067j],
                    [16, 2, 0.001733 + 0.000767j],
                    [17, 2, 0.000000 + 0.000000j],
                    [18, 2, 0.000133 + 0.000067j],
                    [19, 2, 0.000000 + 0.000000j],
                    [20, 2, 0.000000 + 0.000000j],
                    [21, 2, 0.000000 + 0.000000j],
                    [22, 2, 0.015000 + 0.007500j],
                    [23, 2, 0.000500 + 0.000233j],
                    [24, 2, 0.000067 + 0.000033j],
                    [25, 2, 0.001067 + 0.000567j],
                    [26, 2, 0.000000 + 0.000000j],
                    [27, 2, 0.013800 + 0.010667j],
                    [28, 2, 0.001500 + 0.000767j],
                    [29, 2, 0.002767 + 0.001967j],
                    [30, 2, 0.006867 + 0.004033j],
                    [31, 2, 0.002733 + 0.001433j],
                    [32, 2, 0.002233 + 0.001367j],
                    [33, 2, 0.000000 + 0.000000j],
                    [34, 2, 0.000933 + 0.000467j]
                    ])
    bus[:,2] *= load
    # line format: from bus, to bus, line impedance z = r+jx (in p.u.)
    line = np.array([[1, 2, 0.013241, 0.009851],
                     [2, 3, 0.008879, 0.006605],
                     [3, 4, 0.165415, 0.123056],
                     [4, 5, 0.074344, 0.039506],
                     [4, 6, 0.192463, 0.143177],
                     [6, 7, 0.152584, 0.113511],
                     [7, 8, 0.000077, 0.000039],
                     [9, 10, 0.021935, 0.011639],
                     [9, 13, 0.079066, 0.039350],
                     [10, 11, 0.617640, 0.327739],
                     [11, 12, 0.176249, 0.093523],
                     [13, 14, 0.038812, 0.020624],
                     [13, 15, 0.006505, 0.003237],
                     [15, 16, 0.158286, 0.078778],
                     [16, 17, 0.004027, 0.002004],
                     [20, 23, 0.037945, 0.018885],
                     [20, 21, 0.570000, 1.224000],
                     [25, 30, 0.015643, 0.007785],
                     [25, 26, 0.002168, 0.001079],
                     [31, 32, 0.006660, 0.003315],
                     [31, 33, 0.002168, 0.001079],
                     [26, 27, 0.010454, 0.005203],
                     [27, 28, 0.028188, 0.014029],
                     [28, 29, 0.004104, 0.002043],
                     [8, 9, 0.002401, 0.001195],
                     [19, 20, 0.000077, 0.000039],
                     [17, 18, 0.298836, 0.158798],
                     [17, 19, 0.285210, 0.141947],
                     [23, 24, 0.020780, 0.011027],
                     [23, 25, 0.045147, 0.022469],
                     [30, 31, 0.020754, 0.010329],
                     [33, 34, 0.042794, 0.031648],
                     [21, 22, 1.941741, 1.444504]
                    ])
    # line = np.array([[1, 2, 3.34403, 7.88777],
    #                  [2, 3, 5.00, 300.0],
    #                  [3, 4, 37.560096, 88.482919]
    #                  ])
    v2ref = vref**2  # reference node voltage
    return bus, line, v2ref


def load_flow(action,load):
    # reg_tap = action[0]
    # tsf_tap = {1:action[1], 2:action[2], 3:action[3]}
    # cap_tap = {1:action[4], 2:action[5], 3:action[6]}

    vref = 0.95+action[0]*0.01
    tsf_pos = {10: action[1]*0.01+0.95,  # from node 7 to node 8 line[7]
               25: action[2]*0.01+0.95,  # from node 19 to node 20 line[25]
    }
    cap_pos = {10: 0.02*action[3],   # 100KVA, base 5MVA, at node 27/node[26]
               87: 0.03*action[4],   # 150KVA, base 5MVA, at node 29/node[28]
    }


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

        #if ii==83: #84-1
        if ii in cap_pos.keys():
            Aqv[ii - num_ref_node,ii] = -cap_pos[ii]

    # third linear eqn
    Avv[:num_ref_node, :num_ref_node] = np.identity(num_ref_node)

    # forth linear eqn
    for ij in range(num_edge):
        Alp[ij, ij] = -2 * line[ij, 2]
        Alq[ij, ij] = -2 * line[ij, 3]
        Alv[ij, int(line[ij, 0] - 1)] = 1  # from node， the minus 1 is for matching python indexing convention
        #if ij==118:  #119-1
        if ij in tsf_pos.keys():
            Alv[ij, int(line[ij, 1] - 1)] = -1/tsf_pos[ij]**2 # to node
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


if __name__ == "__main__":
    Pij, Qij, Vi2, Lij, total_loss, iter, convergence_flag = load_flow([10,10,10,0,0],1)
    print(np.sqrt(Vi2))