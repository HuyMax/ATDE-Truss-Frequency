import math
import numpy as np
import numpy.matlib
import random
import scipy.stats as stats

# This is the code of the algorithm in the paper: A novel adaptive 3-stage hybrid teaching-based differential evolution algorithm for frequency-constrained truss designs,
# Structures (2022). Please feel free to use it with an appropriate citation.
# Usage:
# Evaluate_Objective: the objective function
# with input: problem label, population size, a candidate solution vector, and penalty factors
# and output: penalized objective function value of the candidate solution and its constraint violation.

def ATDE(Evaluate_Objective, Label, LB, UB, nDVs, NP, penalty_factor, MaxIter, tol, threshold1, threshold2):
    Iter_Plot = np.zeros((MaxIter, 1), dtype=float)
    nFEAs_Plot = np.zeros((MaxIter, 1), dtype=float)
    fbest_Plot = np.zeros((MaxIter, 1), dtype=float)
    gbest_Plot = np.zeros((MaxIter, 1), dtype=float)

    # Phase 1: Initialization
    x = np.matlib.repmat(LB, NP, 1) + np.random.uniform(0, 1, (NP, nDVs)) * np.matlib.repmat(UB - LB, NP, 1)

    # Evaluate the objective function w.r.t constraints
    f, g = Evaluate_Objective(Label, NP, x, penalty_factor)
    nFEAs = 0 # number of finite element analyses (FEAs)

    sorted_ind = np.argsort(f[:, 0], axis=0)

    ibest = sorted_ind[0]
    ibest2 = sorted_ind[1]
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
        penalty_factor[1] = min(3, penalty_factor[1] + 0.05)

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

                if f[i,0] < fMean_of_3: # condition for teaching based mutation
                    mean_of_3 = (x[R[0], :] + x[R[1], :] + x[R[2], :]) / 3  # Eq. 13
                    TF = 1 + np.random.randint(2, size=(nDVs,))
                    r = a + (b - a) * np.random.random(size=(nDVs,))       # Eq. 15
                    if f[i,0] < fR[R0]:
                        v_m = x[i, :] + r * (x[R[R0], :] - TF * mean_of_3) # Eq. 14a
                    else:
                        v_m = x[R[R0], :] + r * (x[i, :] - TF * mean_of_3) # Eq. 14b

                else:
                    F = a + (b - a) * np.random.random(size=(nDVs,))
                    v_m = x[R[0], :] + F * (x[R[1], :] - x[R[2], :])

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
                    v_m = x[Rbest, :] + F * (x[R[0], :] - x[R[1], :])  # Eq. 17
                else:
                    if i == ibest: Rbest = ibest
                    else: Rbest = np.random.choice([ibest, ibest2])
                    v_m = x[i, :] + F * (x[Rbest,:] - x[i, :]) + F * (x[R[0], :] - x[R[1], :])  # Eq. 16

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
        fnew, gnew = Evaluate_Objective(Label, NP, u, penalty_factor)
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
        xbest = xnew[ibest, :]
        fbest = f[ibest, 0]
        gbest = g[ibest, 0]
        x = xnew
        fmean = np.mean(f)

        Iter_Plot[Iter, 0] = Iter
        nFEAs_Plot[Iter, 0] = nFEAs
        fbest_Plot[Iter, 0] = fbest
        gbest_Plot[Iter, 0] = gbest

        delta = abs(abs(fmean)/abs(fbest) - 1)
        print('Iter: ' + str(Iter) + ', nFEAs: ' + str(nFEAs) + ', fbest: ' + str(fbest) + ', fmean: ' + str(fmean))

        if delta <= tol or Iter == MaxIter: # stopping criteria
            break

    return (xbest, fbest, Iter, nFEAs, Iter_Plot[0:Iter+1, 0], nFEAs_Plot[0:Iter+1, 0], fbest_Plot[0:Iter+1, 0], gbest_Plot[0:Iter+1, 0])
