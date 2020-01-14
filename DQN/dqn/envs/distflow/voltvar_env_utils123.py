import numpy as np

def obtain_network_data(vref=1.0,load=1.0):
    # function: obtain_network_data: a simple data storage function, the format of the data is similar to matpower data
    # IEEE 16 bus data:
    # bus format:   bus num, type, Sd (in p.u.), Sg (in p.u.)
    # note that, Sd is going to be substituted by modified CALLI VALLI data. The values showing here are just
    # sample values (the original IEEE 16 bus values). Note: place all reference buses at the top of the list
    # input: Feeder_size: an integer representing the number of node in the feeder
    bus = np.array([[1, 1, 0.000000 + 0.000000j],
                    [2, 2, 0.002667 + 0.001333j],
                    [3, 2, 0.001333 + 0.000667j],
                    [4, 2, 0.000000 + 0.000000j],
                    [5, 2, 0.002667 + 0.001333j],
                    [6, 2, 0.001333 + 0.000667j],
                    [7, 2, 0.002667 + 0.001333j],
                    [8, 2, 0.001333 + 0.000667j],
                    [9, 2, 0.000000 + 0.000000j],
                    [10, 2, 0.002667 + 0.001333j],
                    [11, 2, 0.001333 + 0.000667j],
                    [12, 2, 0.002667 + 0.001333j],
                    [13, 2, 0.001333 + 0.000667j],
                    [14, 2, 0.000000 + 0.000000j],
                    [15, 2, 0.000000 + 0.000000j],
                    [16, 2, 0.000000 + 0.000000j],
                    [17, 2, 0.002667 + 0.001333j],
                    [18, 2, 0.001333 + 0.000667j],
                    [19, 2, 0.000000 + 0.000000j],
                    [20, 2, 0.002667 + 0.001333j],
                    [21, 2, 0.002667 + 0.001333j],
                    [22, 2, 0.000000 + 0.000000j],
                    [23, 2, 0.002667 + 0.001333j],
                    [24, 2, 0.000000 + 0.000000j],
                    [25, 2, 0.002667 + 0.001333j],
                    [26, 2, 0.000000 + 0.000000j],
                    [27, 2, 0.000000 + 0.000000j],
                    [28, 2, 0.000000 + 0.000000j],
                    [29, 2, 0.002667 + 0.001333j],
                    [30, 2, 0.002667 + 0.001333j],
                    [31, 2, 0.002667 + 0.001333j],
                    [32, 2, 0.001333 + 0.000667j],
                    [33, 2, 0.001333 + 0.000667j],
                    [34, 2, 0.002667 + 0.001333j],
                    [35, 2, 0.002667 + 0.001333j],
                    [36, 2, 0.002667 + 0.001333j],
                    [37, 2, 0.000000 + 0.000000j],
                    [38, 2, 0.002667 + 0.001333j],
                    [39, 2, 0.001333 + 0.000667j],
                    [40, 2, 0.001333 + 0.000667j],
                    [41, 2, 0.000000 + 0.000000j],
                    [42, 2, 0.001333 + 0.000667j],
                    [43, 2, 0.001333 + 0.000667j],
                    [44, 2, 0.002667 + 0.001333j],
                    [45, 2, 0.000000 + 0.000000j],
                    [46, 2, 0.001333 + 0.000667j],
                    [47, 2, 0.001333 + 0.000667j],
                    [48, 2, 0.007000 + 0.005000j],
                    [49, 2, 0.014000 + 0.010000j],
                    [50, 2, 0.009333 + 0.006333j],
                    [51, 2, 0.002667 + 0.001333j],
                    [52, 2, 0.001333 + 0.000667j],
                    [53, 2, 0.002667 + 0.001333j],
                    [54, 2, 0.002667 + 0.001333j],
                    [55, 2, 0.000000 + 0.000000j],
                    [56, 2, 0.001333 + 0.000667j],
                    [57, 2, 0.001333 + 0.000667j],
                    [58, 2, 0.000000 + 0.000000j],
                    [59, 2, 0.001333 + 0.000667j],
                    [60, 2, 0.001333 + 0.000667j],
                    [61, 2, 0.001333 + 0.000667j],
                    [62, 2, 0.000000 + 0.000000j],
                    [63, 2, 0.002667 + 0.001333j],
                    [64, 2, 0.002667 + 0.001333j],
                    [65, 2, 0.005000 + 0.002333j],
                    [66, 2, 0.009333 + 0.006667j],
                    [67, 2, 0.005000 + 0.002333j],
                    [68, 2, 0.000000 + 0.000000j],
                    [69, 2, 0.001333 + 0.000667j],
                    [70, 2, 0.002667 + 0.001333j],
                    [71, 2, 0.001333 + 0.000667j],
                    [72, 2, 0.002667 + 0.001333j],
                    [73, 2, 0.000000 + 0.000000j],
                    [74, 2, 0.002667 + 0.001333j],
                    [75, 2, 0.002667 + 0.001333j],
                    [76, 2, 0.002667 + 0.001333j],
                    [77, 2, 0.016333 + 0.012000j],
                    [78, 2, 0.002667 + 0.001333j],
                    [79, 2, 0.000000 + 0.000000j],
                    [80, 2, 0.002667 + 0.001333j],
                    [81, 2, 0.002667 + 0.001333j],
                    [82, 2, 0.000000 + 0.000000j],
                    [83, 2, 0.002667 + 0.001333j],
                    [84, 2, 0.001333 + 0.000667j],
                    [85, 2, 0.001333 + 0.000667j],
                    [86, 2, 0.002667 + 0.001333j],
                    [87, 2, 0.001333 + 0.000667j],
                    [88, 2, 0.002667 + 0.001333j],
                    [89, 2, 0.002667 + 0.001333j],
                    [90, 2, 0.000000 + 0.000000j],
                    [91, 2, 0.002667 + 0.001333j],
                    [92, 2, 0.000000 + 0.000000j],
                    [93, 2, 0.002667 + 0.001333j],
                    [94, 2, 0.000000 + 0.000000j],
                    [95, 2, 0.002667 + 0.001333j],
                    [96, 2, 0.001333 + 0.000667j],
                    [97, 2, 0.001333 + 0.000667j],
                    [98, 2, 0.000000 + 0.000000j],
                    [99, 2, 0.002667 + 0.001333j],
                    [100, 2, 0.002667 + 0.001333j],
                    [101, 2, 0.002667 + 0.001333j],
                    [102, 2, 0.000000 + 0.000000j],
                    [103, 2, 0.001333 + 0.000667j],
                    [104, 2, 0.002667 + 0.001333j],
                    [105, 2, 0.002667 + 0.001333j],
                    [106, 2, 0.000000 + 0.000000j],
                    [107, 2, 0.002667 + 0.001333j],
                    [108, 2, 0.002667 + 0.001333j],
                    [109, 2, 0.000000 + 0.000000j],
                    [110, 2, 0.002667 + 0.001333j],
                    [111, 2, 0.000000 + 0.000000j],
                    [112, 2, 0.001333 + 0.000667j],
                    [113, 2, 0.001333 + 0.000667j],
                    [114, 2, 0.002667 + 0.001333j],
                    [115, 2, 0.001333 + 0.000667j],
                    [116, 2, 0.000000 + 0.000000j],
                    [117, 2, 0.000000 + 0.000000j],
                    [118, 2, 0.000000 + 0.000000j],
                    [119, 2, 0.002667 + 0.001333j],
                    [120, 2, 0.000000 + 0.000000j]
                    ])
    bus[:,2] *= load
    # line format: from bus, to bus, line impedance z = r+jx (in p.u.)
    line = np.array([[2, 3, 0.038186, 0.038711],
                    [2, 4, 0.054551, 0.055302],
                    [2, 8, 0.015073, 0.030880],
                    [4, 5, 0.043641, 0.044241],
                    [4, 6, 0.070916, 0.071892],
                    [6, 7, 0.054551, 0.055302],
                    [8, 9, 0.010049, 0.020587],
                    [9, 13, 0.049096, 0.049772],
                    [9, 10, 0.049096, 0.049772],
                    [9, 14, 0.015073, 0.030880],
                    [10, 15, 0.092736, 0.094013],
                    [14, 35, 0.032730, 0.033181],
                    [14, 19, 0.041452, 0.084921],
                    [15, 12, 0.054551, 0.055302],
                    [15, 11, 0.054551, 0.055302],
                    [16, 17, 0.081826, 0.082953],
                    [16, 18, 0.076371, 0.077423],
                    [19, 20, 0.054551, 0.055302],
                    [19, 22, 0.015073, 0.030880],
                    [20, 21, 0.070916, 0.071892],
                    [22, 23, 0.114557, 0.116134],
                    [22, 24, 0.012561, 0.025734],
                    [24, 25, 0.120012, 0.121664],
                    [24, 26, 0.013817, 0.028307],
                    [26, 27, 0.017585, 0.039452],
                    [26, 29, 0.010049, 0.020587],
                    [27, 28, 0.013816, 0.030998],
                    [27, 32, 0.049096, 0.049772],
                    [28, 34, 0.109102, 0.110604],
                    [29, 30, 0.015073, 0.030880],
                    [30, 31, 0.017586, 0.036027],
                    [31, 116, 0.010049, 0.020587],
                    [32, 33, 0.065461, 0.066362],
                    [35, 16, 0.021820, 0.022121],
                    [36, 37, 0.032657, 0.073269],
                    [36, 41, 0.012561, 0.025734],
                    [37, 38, 0.065461, 0.066362],
                    [37, 39, 0.054551, 0.055302],
                    [39, 40, 0.070916, 0.071892],
                    [41, 42, 0.070916, 0.071892],
                    [41, 43, 0.012561, 0.025734],
                    [43, 44, 0.109102, 0.110604],
                    [43, 45, 0.010049, 0.020587],
                    [45, 46, 0.043641, 0.044241],
                    [45, 48, 0.012561, 0.025734],
                    [46, 47, 0.065461, 0.066362],
                    [48, 49, 0.007537, 0.015440],
                    [48, 50, 0.012561, 0.025734],
                    [50, 51, 0.012561, 0.025734],
                    [51, 52, 0.012561, 0.025734],
                    [52, 117, 0.025122, 0.051467],
                    [53, 54, 0.010049, 0.020587],
                    [54, 55, 0.006281, 0.012867],
                    [55, 56, 0.013817, 0.028307],
                    [55, 58, 0.017586, 0.036027],
                    [56, 57, 0.013817, 0.028307],
                    [58, 59, 0.054551, 0.055302],
                    [58, 61, 0.037683, 0.077201],
                    [59, 60, 0.054551, 0.055302],
                    [61, 62, 0.027634, 0.056614],
                    [61, 63, 0.051913, 0.009543],
                    [63, 64, 0.036339, 0.006680],
                    [64, 65, 0.072679, 0.013361],
                    [65, 66, 0.088253, 0.016224],
                    [66, 67, 0.067487, 0.012406],
                    [68, 69, 0.043641, 0.044241],
                    [68, 73, 0.013817, 0.028307],
                    [68, 98, 0.012561, 0.025734],
                    [69, 70, 0.060006, 0.060832],
                    [70, 71, 0.070916, 0.071892],
                    [71, 72, 0.060006, 0.060832],
                    [73, 74, 0.060006, 0.060832],
                    [73, 77, 0.010049, 0.020587],
                    [74, 75, 0.076371, 0.077423],
                    [75, 76, 0.087281, 0.088483],
                    [77, 78, 0.020098, 0.041174],
                    [77, 87, 0.035171, 0.072054],
                    [78, 79, 0.005024, 0.010293],
                    [79, 80, 0.011305, 0.023160],
                    [79, 81, 0.023866, 0.048894],
                    [81, 82, 0.023866, 0.048894],
                    [82, 83, 0.012561, 0.025734],
                    [82, 85, 0.147287, 0.149315],
                    [83, 84, 0.012561, 0.025734],
                    [85, 86, 0.103646, 0.105073],
                    [87, 88, 0.022610, 0.046321],
                    [88, 89, 0.038186, 0.038711],
                    [88, 90, 0.013817, 0.028307],
                    [90, 91, 0.049096, 0.049772],
                    [90, 92, 0.011305, 0.023160],
                    [92, 93, 0.065461, 0.066362],
                    [92, 94, 0.011305, 0.023160],
                    [94, 95, 0.060006, 0.060832],
                    [94, 96, 0.015073, 0.030880],
                    [96, 97, 0.043641, 0.044241],
                    [98, 99, 0.013817, 0.028307],
                    [99, 100, 0.027634, 0.056614],
                    [100, 101, 0.015073, 0.030880],
                    [101, 120, 0.040195, 0.082348],
                    [102, 103, 0.049096, 0.049772],
                    [102, 106, 0.013817, 0.028307],
                    [103, 104, 0.070916, 0.071892],
                    [104, 105, 0.152742, 0.154845],
                    [106, 107, 0.049096, 0.049772],
                    [106, 109, 0.016329, 0.033454],
                    [107, 108, 0.125467, 0.127194],
                    [109, 110, 0.098191, 0.099543],
                    [109, 119, 0.050244, 0.102935],
                    [110, 111, 0.065461, 0.066362],
                    [111, 112, 0.125467, 0.127194],
                    [111, 113, 0.027275, 0.027651],
                    [113, 114, 0.114557, 0.116134],
                    [114, 115, 0.070916, 0.071892],
                    [19, 36, 0.018842, 0.038600],
                    [1, 2, 0.020098, 0.041174],
                    [14, 53, 0.020098, 0.041174],
                    [61, 68, 0.017586, 0.036027],
                    [98, 102, 0.012561, 0.025734],
                    [62, 118, 1.270000, 2.720000]
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
    tsf_pos = {10: action[1]*0.01+0.95,  # from node 10 to node 15 line[10]
               24: action[2]*0.01+0.95,  # from node 26 to node 27 line[24]
               116: action[3]*0.01+0.95  # from node 61 to node 68 line[116]
    }
    cap_pos = {10: 0.04*action[4],   # 200KVA, base 5MVA, at node 11/node[10]
               87: 0.01*action[5],   # 50KVA, base 5MVA, at node 88/node[87]
               89: 0.01*action[6],   # 50KVA, base 5MVA, at node 90/node[89]
               91: 0.01*action[7]   # 50KVA, base 5MVA, at node 92/node[91]
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
    Pij, Qij, Vi2, Lij, total_loss, iter, convergence_flag = load_flow([0,0,0,0,0,0,0,0],1)
    print(np.sqrt(Vi2))