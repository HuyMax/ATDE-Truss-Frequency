import math
import numpy as np
import numpy.matlib
import random
import scipy.stats as stats

# Notes:
# *** -> Check when changing problems

def ATDE(Evaluate_Objective, Label, LB, UB, nDVs, NP, penal, MaxIter, tol, threshold1, threshold2):
    Iter_Plot = np.zeros((MaxIter, 1), dtype=float)
    nFEAs_Plot = np.zeros((MaxIter, 1), dtype=float)
    fbest_Plot = np.zeros((MaxIter, 1), dtype=float)
    gbest_Plot = np.zeros((MaxIter, 1), dtype=float)

    # Phase 1: Initialization
    x = np.matlib.repmat(LB, NP, 1) + np.random.uniform(0, 1, (NP, nDVs)) * np.matlib.repmat(UB - LB, NP, 1)

    # Evaluate the objective function w.r.t constraints
    f, g = Evaluate_Objective(Label, NP, x, penal)
    nFEAs = 0 # number of finite element analyses (FEAs)

    sorted_ind = np.argsort(f[:, 0], axis=0)

    ibest = sorted_ind[0]
    ibest2 = sorted_ind[1]
    ibest3 = sorted_ind[2]
    xbest = x[ibest,:]
    fbest = f[ibest,0]
    gbest = g[ibest,0]
    fmean = np.mean(f)

    delta = abs(abs(fmean)/abs(fbest) - 1)

    # Mutant vector
    v = np.zeros((NP, nDVs), dtype=float)
    # Trial vector
    u = np.zeros((NP, nDVs), dtype=float)
    a, b, TC = 0.5, 1, 100
    step = (b - a) / (2*TC)
    for Iter in range(MaxIter):
        # *** Update penalty ***
        penal[1] = min(3, penal[1] + 0.05)

        if Iter < TC:
            a, b = a+step, b-step

        for i in range(NP):
            # Phase 2: Mutation
            if delta > threshold1:
                candidates = list(range(0, NP))
                candidates.remove(i)
                R = np.asarray(random.sample(candidates, 3))
                fR = f[R, 0]
                fMean_of_3 = np.mean(fR)
                R0 = np.argmin(fR, axis=0)

                if f[i,0] < fMean_of_3: # i-th student is better than mean of 3 random students
                    mean_of_3 = (x[R[0], :] + x[R[1], :] + x[R[2], :]) / 3  # mean student of 3 students
                    TF = 1 + np.random.randint(2, size=(nDVs,))
                    r = a + (b - a) * np.random.random(size=(nDVs,))
                    if f[i,0] < fR[R0]:
                        v_m = x[i, :] + r * (x[R[R0], :] - TF * mean_of_3)
                    else:
                        v_m = x[R[R0], :] + r * (x[i, :] - TF * mean_of_3)

                else:
                    F = a + (b - a) * np.random.random(size=(nDVs,))
                    v_m = x[R[0], :] + F * (x[R[1], :] - x[R[2], :]) # rand/1

            else:
                candidates = list(range(0, NP))
                candidates.remove(i)
                if i != ibest:
                    candidates.remove(ibest)
                if i != ibest2:
                    candidates.remove(ibest2)
                R = np.asarray(random.sample(candidates, 2))
                Rbest = np.random.choice([ibest, ibest2])
                F = a + (b - a) * np.random.random(size=(nDVs,))

                if delta <= threshold2:
                    v_m = x[Rbest, :] + F * (x[R[0], :] - x[R[1], :])  # 2best/1
                else:
                    if i == ibest: Rbest = ibest
                    else: Rbest = np.random.choice([ibest, ibest2])
                    v_m = x[i, :] + F * (x[Rbest,:] - x[i, :]) + F * (x[R[0], :] - x[R[1], :])  # current-to-2best/1

            # Return design variables violated bounds to the search space
            v_m = (v_m >= LB)*v_m + (v_m < LB)*(2*LB - v_m)
            v_m = (UB >= v_m)*v_m + (UB < v_m)*(2*UB - v_m)
            v_m = (v_m >= LB)*v_m + (v_m < LB)*LB
            v_m = (UB >= v_m)*v_m + (UB < v_m)*UB

            v[i, :] = v_m

            # Phase 3: Crossover
            K = random.sample(list(range(0, nDVs)), 1)
            Cr = 0.7 + (1-0.7)*np.random.random()
            t = np.random.uniform(0, 1, nDVs) <= Cr
            t[K] = 1
            u[i, :] = t*v[i, :] + (1-t)*x[i, :]

        # Phase 4: Selection
        fnew, gnew = Evaluate_Objective(Label, NP, u, penal)
        nFEAs += NP

        # elitist
        f_temp = np.concatenate((f,fnew))
        g_temp = np.concatenate((g,gnew))
        x_temp = np.concatenate((x,u))
        sorted_index = np.argsort(f_temp[:,0])
        xnew = x_temp[sorted_index[0:NP],:]
        f = f_temp[sorted_index[0:NP]]
        g = g_temp[sorted_index[0:NP]]

        sorted_ind = np.argsort(f[:, 0], axis=0)
        ibest = sorted_ind[0]
        ibest2 = sorted_ind[1]
        ibest3 = sorted_ind[2]
        xbest = xnew[ibest, :]
        fbest = f[ibest, 0]
        gbest = g[ibest, 0]
        x = xnew
        fmean = np.mean(f)

        Iter_Plot[Iter, 0] = Iter
        nFEAs_Plot[Iter, 0] = nFEAs
        fbest_Plot[Iter, 0] = fbest
        gbest_Plot[Iter, 0] = gbest

        delta = abs(abs(fmean)/abs(fbest) - 1) # for truss
        print('Iter: ' + str(Iter) + ', nFEAs: ' + str(nFEAs) + ', fbest: ' + str(fbest) + ', fmean: ' + str(fmean))

        if delta <= tol or Iter == MaxIter: # for truss
            break

    return (xbest, fbest, Iter, nFEAs, Iter_Plot[0:Iter+1, 0], nFEAs_Plot[0:Iter+1, 0], fbest_Plot[0:Iter+1, 0], gbest_Plot[0:Iter+1, 0])