
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.stats.mstats import gmean
from time import time
from multiprocessing import Process, Pipe
import sys, os, warnings
from a2dr.precondition import precondition
from a2dr.acceleration import aa_weights
from a2dr.utilities import get_version


sys_stdout_origin = sys.stdout



def map_g(p_list,A,b,v,t,dk,n_list_cumsum):
    N=len(p_list)
    #
    v_list = [v[n_list_cumsum[i]:n_list_cumsum[i+1]]  for i in range(N) ]
    x_list = [p_list[i](v_list[i],t) for i in range(N)]
    x_half = np.concatenate(x_list, axis=0)
    v_half = 2*x_half-v
    dk = sp.linalg.lsqr(A, A.dot(v_half) - b, atol=1e-10, btol=1e-10, x0=dk)[0]
    x_new = v_half - dk
    f = x_new - x_half
    v_new = v + f
    return v_new, f, dk, x_half, x_half - v


def lmaa(p_list, A_list=[], b=np.array([]), v_init=None, n_list=None, *args, **kwargs):
    start = time()

    # Problem parameters.
    max_iter = kwargs.pop("max_iter", 1000)
    t_init = kwargs.pop("t_init", 1 / 10)  # Step size.
    eps_abs = kwargs.pop("eps_abs", 1e-6)  # Absolute stopping tolerance.
    eps_rel = kwargs.pop("eps_rel", 1e-8)  # Relative stopping tolerance.
    precond = kwargs.pop("precond", True)  # Precondition A and b?
    ada_reg = kwargs.pop("ada_reg", True)  # Adaptive regularization?

    # AA-II parameters.
    anderson = kwargs.pop("anderson", True)
    m_accel = int(kwargs.pop("m_accel", 10))  # Maximum past iterations to keep (>= 0).
    lam_accel = kwargs.pop("lam_accel", 1e-8)  # AA-II regularization weight.
    aa_method = kwargs.pop("aa_method", "lstsq")  # Algorithm for solving AA LS problem.

    # Safeguarding parameters.
    D_safe = kwargs.pop("D_safe", 1e6)
    eps_safe = kwargs.pop("eps_safe", 1e-6)
    M_safe = kwargs.pop("M_safe", int(max_iter / 100))

    c = kwargs.pop("c", 1-1e-6)
    # Printout parameters
    verbose = kwargs.pop("verbose", True)

    # Validate parameters.
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if t_init <= 0:
        raise ValueError("t_init must be a positive scalar.")
    if eps_abs < 0:
        raise ValueError("eps_abs must be a non-negative scalar.")
    if eps_rel < 0:
        raise ValueError("eps_rel must be a non-negative scalar.")
    if m_accel <= 0:
        raise ValueError("m_accel must be a positive integer.")
    if lam_accel < 0:
        raise ValueError("lam_accel must be a non-negative scalar.")
    if not aa_method in ["lstsq", "lsqr"]:
        raise ValueError("aa_method must be either 'lstsq' or 'lsqr'.")
    # if D_safe < 0:
    #     raise ValueError("D_safe must be a non-negative scalar.")
    # if eps_safe < 0:
    #     raise ValueError("eps_safe must be a non-negative scalar.")
    # if M_safe <= 0:
    #     raise ValueError("M_safe must be a positive integer.")

    # DRS parameters.
    N = len(p_list)  # Number of subproblems.
    has_constr = len(A_list) != 0
    if len(A_list) == 0:
        if b.size != 0:
            raise ValueError("Dimension mismatch: nrow(A_i) != nrow(b)")
        if n_list is not None:
            if len(n_list) != N:
                raise ValueError("n_list must have exactly {} entries".format(N))
            A_list = [sp.csr_matrix((0, ni)) for ni in n_list]
        elif v_init is not None:
            if len(v_init) != N:
                raise ValueError("v_init must be None or contain exactly {} entries".format(N))
            A_list = [sp.csr_matrix((0, vi.shape[0])) for vi in v_init]
        else:
            raise ValueError("n_list or v_init must be defined if A_list and b are empty")
    if len(A_list) != N:
        raise ValueError("A_list must be empty or contain exactly {} entries".format(N))
    if v_init is None:
        # v_init = [np.random.randn(A.shape[1]) for A in A_list]
        v_init = [np.zeros(A.shape[1]) for A in A_list]
        # v_init = [sp.csc_matrix((A.shape[1],1)) for A in A_list]
    if len(v_init) != N:
        raise ValueError("v_init must be None or contain exactly {} entries".format(N))

    # Variable size list.
    if n_list is None:
        n_list = [A_list[i].shape[1] for i in range(N)]
    if len(n_list) != N:
        raise ValueError("n_list must be None or contain exactly {} entries".format(N))
    n_list_cumsum = np.insert(np.cumsum(n_list), 0, 0)

    for i in range(N):
        if A_list[i].shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch: nrow(A_i) != nrow(b)")
        elif A_list[i].shape[1] != v_init[i].shape[0]:
            raise ValueError("Dimension mismatch: ncol(A_i) != nrow(v_i)")
        elif A_list[i].shape[1] != n_list[i]:
            raise ValueError("Dimension mismatch: ncol(A_i) != n_i")
        if not sp.issparse(A_list[i]):
            A_list[i] = sp.csr_matrix(A_list[i])

    if verbose:
        version = get_version("__init__.py")
        line_solver = "a2dr v" + version + " - Prox-Affine Distributed Convex Optimization Solver"
        dashes = "-" * len(line_solver)
        ddashes = "=" * len(line_solver)
        line_authors = "(c) Anqi Fu, Junzi Zhang"
        num_spaces_authors = (len(line_solver) - len(line_authors)) // 2
        line_affil = "Stanford University   2019"
        num_spaces_affil = (len(line_solver) - len(line_affil)) // 2
        print(dashes)
        print(line_solver)
        print(" " * num_spaces_authors + line_authors)
        print(" " * num_spaces_affil + line_affil)
        print(dashes)

    # Precondition data.
    if precond and has_constr:
        if verbose:
            print('### Preconditioning starts ... ###')
        p_list, A_list, b, e_pre = precondition(p_list, A_list, b)
        t_init = 1 / gmean(e_pre) ** 2 / 10
        if verbose:
            print('### Preconditioning finished.  ###')

    sigma0 = kwargs.pop("sigma0", 1e-10)
    sigma1 = kwargs.pop("sigma1", 1)
    c=max((t_init*sigma1-1)/(t_init*sigma1+1),(1-t_init*sigma0)/(1+t_init*sigma0))
    c=np.sqrt((3+c**2))/2
    if verbose:
        print("max_iter = {}, t_init (after preconditioning) = {:.2f}".format(
            max_iter, t_init))
        print("eps_abs = {:.2e}, eps_rel = {:.2e}, precond = {!r}".format(
            eps_abs, eps_rel, precond))
        print("ada_reg = {!r}, anderson = {!r}, m_accel = {}".format(
            ada_reg, anderson, m_accel))
        print("lam_accel = {:.2e}, aa_method = {}, D_safe = {:.2e}".format(
            lam_accel, aa_method, D_safe))
        print("eps_safe = {:.2e}, M_safe = {:d}".format(
            eps_safe, M_safe))

    # Store constraint matrix for projection step.
    A = sp.csr_matrix(sp.hstack(A_list))
    if verbose:
        print("variables n = {}, constraints m = {}".format(A.shape[1], A.shape[0]))
        print("nnz(A) = {}".format(A.nnz))
        print("Setup time: {:.2e}".format(time() - start))

    # Check linear feasibility
    sys.stdout = open(os.devnull, 'w')
    r1norm = sp.linalg.lsqr(A, b)[3]
    sys.stdout.close()
    sys.stdout = sys_stdout_origin
    if r1norm >= np.sqrt(eps_abs):  # infeasible
        if verbose:
            print('Infeasible linear equality constraint: minimum constraint violation = {:.2e}'.format(r1norm))
            print('Status: Terminated due to linear infeasibility')
            print("Solve time: {:.2e}".format(time() - start))
        return {"x_vals": None, "primal": None, "dual": None, "num_iters": None, "solve_time": None}

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("----------------------------------------------------")

    # Set up the workers.


    # Initialize AA-II variables.
    if anderson:  # TODO: Store and update these efficiently as arrays.
        n_sum = np.sum([np.prod(v.shape) for v in v_init])
        g_vec = np.zeros(n_sum)  # g^(k) = v^(k) - F(v^(k)).
        s_hist = []  # History of s^(j) = v^(j+1) - v^(j), kept in S^(k) = [s^(k-m_k) ... s^(k-1)].
        y_hist = []  # History of y^(j) = g^(j+1) - g^(j), kept in Y^(k) = [y^(k-m_k) ... y^(k-1)].
        n_AA = M_AA = 0  # Safeguarding counters.

    # A2DR loop.
    k = 0
    finished = False
    safeguard = True
    r_primal = np.zeros(max_iter)
    r_dual = np.zeros(max_iter)
    r_dr = np.zeros(max_iter)
    time_iter = np.zeros(max_iter)
    r_best = np.inf

    # Warm start terms.
    dk = np.zeros(A.shape[1])
    sol = np.zeros(A.shape[0])

    v =np.concatenate(v_init)

    f_list = np.zeros((A.shape[1],m_accel+1))
    g_list = np.zeros((A.shape[1],m_accel+1))
    x_list = np.zeros((A.shape[1],m_accel+1))
    F_norm = np.zeros((m_accel+1))
    M = np.zeros((m_accel+1,m_accel+1))
    idx = 0
    eta0 = 2
    eta1 = 0.25
    mu = 1e-8
    delta = 2
    p1 = 0.01
    p2 = 0.25
    curr_dk = dk.copy()
    while not finished:
        if k==0:
            x_list[:,0] = v
            curr_g, curr_f, curr_dk, curr_x_half, curr_xvdiff =map_g(p_list,A,b,v,t_init,curr_dk,n_list_cumsum)
            g_list[:,0] = curr_g
            F_norm[0] = np.sum(curr_f**2)
            f_list[:, 0] = curr_f/np.sqrt(F_norm[0])
            M[0,0] = 1
            Ax_half = A.dot(curr_x_half)
            r_primal_vec = (Ax_half) - b
            r_primal[k] = LA.norm(r_primal_vec, ord=2)
            subgrad = curr_xvdiff / t_init
            # sol = LA.lstsq(A.T, subgrad, rcond=None)[0]
            sys.stdout = open(os.devnull, 'w')
            sol = sp.linalg.lsqr(A.T, subgrad, atol=1e-10, btol=1e-10, x0=sol)[0]
            sys.stdout.close()
            sys.stdout = sys_stdout_origin
            r_dual_vec = A.T.dot(sol) - subgrad
            r_dual[k] = LA.norm(r_dual_vec, ord=2)
            # Save x_i^(k+1/2) if residual norm is smallest so far.
            r_all = LA.norm(np.concatenate([r_primal_vec, r_dual_vec]), ord=2)
            # Store ||r^0||_2 for stopping criterion.
            r_all_0 = r_all
            x_final = curr_x_half
            r_best = r_all
            k_best = k
            r_dr[k] = np.sqrt(F_norm[0])
            time_iter[k] = time() - start
            v = curr_g.copy()
            curr_g, curr_f, curr_dk, curr_x_half, curr_xvdiff = map_g(p_list,A,b,v,t_init,curr_dk,n_list_cumsum)
            idx += 1
            m_hat = min(idx, m_accel)
            id = np.mod(idx,m_accel+1)
            x_list[:,id] = v
            g_list[:,id] = curr_g
            F_norm[id] = np.sum(curr_f ** 2)
            f_list[:, id] = curr_f / np.sqrt(F_norm[id])
            k += 1
            continue
        if id>0:
            M[0:id,id] = (f_list[:,0:id]).T @ f_list[:,id]
        if id<m_hat:
            M[id+1:m_hat+1,id]= (f_list[:,id+1:m_hat+1]).T @ f_list[:,id]
        M[id,id] = 1
        M[id, 0:m_hat+1] = (M[0:m_hat+1,id]).T
        k0=np.argmin(F_norm[0:m_hat+1])
        normal_Fnorm= np.sqrt(F_norm[0:m_hat+1])/np.sqrt(F_norm[k0])
        lam = mu
        tM = np.diag(normal_Fnorm)@ M[0:m_hat+1,0:m_hat+1] @ np.diag(normal_Fnorm)
        bb = tM[:,k0]
        B = np.repeat(np.array([bb]).T,m_hat+1,axis=1)
        D = tM + 1 - B - B.T
        D = np.delete(D, k0, axis=0)
        D = np.delete(D, k0, axis=1)
        tb = 1 - bb
        tb = np.delete(tb,k0)
        tA = D + lam*np.eye(m_hat)
        alpha = np.linalg.lstsq(tA,tb,rcond=None)[0]

        gamma = 1e-4*np.ones((m_hat+1));
        gamma[k0] =1 - 1e-4 * (m_hat);

        sum_f =np.dot( F_norm[0:m_hat+1] , gamma)

        g_k0=g_list[:,k0]

        temp_alpha= np.concatenate((alpha[0:k0],[1-np.sum(alpha)],alpha[k0:]))
        g_hat= np.dot(g_list[:,0:m_hat+1],temp_alpha)
        descent = alpha.T @ D @ alpha -2 * np.dot(tb, alpha)
        normf_hat = F_norm[k0]*(1+descent)

        trial_g, trial_f, trial_dk, trial_x_half, trial_xvdiff=map_g(p_list,A,b,g_hat,t_init,curr_dk,n_list_cumsum)
        trial_Fnorm = np.sum(trial_f**2)
        pred = sum_f - c*c*normf_hat
        ared = sum_f - trial_Fnorm
        rho = ared/pred
        if rho<p1:
            mu = eta0 * mu
        elif rho>p2:
            mu = eta1 * mu

        if rho<p1:
            v = g_k0.copy()
            curr_g, curr_f, curr_dk, curr_x_half, curr_xvdiff = map_g(p_list,A,b,v,t_init,curr_dk,n_list_cumsum)
            idx = idx + 1
            m_hat = min(idx, m_accel)
            id = np.mod(idx, m_accel+1)
            x_list[:,id] = v
            g_list[:,id] = curr_g
            F_norm[id] = np.sum(curr_f**2)
            f_list[:,id] = curr_f/(np.sqrt(F_norm[id]))
        else:
            v = g_hat.copy()
            curr_g = trial_g.copy()
            curr_f = trial_f.copy()
            curr_dk, curr_x_half, curr_xvdiff = trial_dk.copy(),trial_x_half.copy(),trial_xvdiff.copy()
            idx = idx + 1
            m_hat = min(idx, m_accel)
            id = np.mod(idx, m_accel + 1)
            x_list[:, id] = v
            g_list[:, id] = curr_g
            F_norm[id] = np.sum(curr_f ** 2)
            f_list[:, id] = trial_f / (np.sqrt(F_norm[id]))
        # k = k + 1
        # Gather v_i1^(k+1/2) from nodes.
        # Projection step for x^(k+1).

        # Compute l2-norm of primal and dual residuals.
        Ax_half = A.dot(curr_x_half)
        r_primal_vec = (Ax_half) - b
        r_primal[k] = LA.norm(r_primal_vec, ord=2)
        r_dr[k] = np.sqrt(F_norm[id])
        time_iter[k]=time()-start
        subgrad = curr_xvdiff / t_init
        # sol = LA.lstsq(A.T, subgrad, rcond=None)[0]
        sys.stdout = open(os.devnull, 'w')
        sol = sp.linalg.lsqr(A.T, subgrad, atol=1e-10, btol=1e-10, x0=sol)[0]
        sys.stdout.close()
        sys.stdout = sys_stdout_origin
        r_dual_vec = A.T.dot(sol) - subgrad
        r_dual[k] = LA.norm(r_dual_vec, ord=2)
        # Save x_i^(k+1/2) if residual norm is smallest so far.
        r_all = LA.norm(np.concatenate([r_primal_vec, r_dual_vec]), ord=2)
        # Store ||r^0||_2 for stopping criterion.
        if  r_all < r_best:
            x_final = curr_x_half
            r_best = r_all
            k_best = k

        # Save x_i^(k+1/2) if residual norm is smallest so far.
        if (k % 100 == 0 or k == max_iter - 1) and verbose:
            # print every 100 iterations or reaching maximum
            print("{}| {}  {}  {}  {}".format(str(k).rjust(6),
                                              format(r_all, ".2e").ljust(10),
                                              format(r_primal[k], ".2e").ljust(11),
                                              format(r_dual[k], ".2e").ljust(9),
                                              format(time() - start, ".2e").ljust(8)))

        # Stop when residual norm falls below tolerance.
        k = k + 1
        finished = k >= max_iter or (r_all <= eps_abs + eps_rel * r_all_0)
        if r_all <= eps_abs + eps_rel * r_all_0 and k % 100 != 0 and verbose:
            # print the best iterate
            print("{}| {}  {}  {}  {}".format(str(k - 1).rjust(6),
                                              format(r_all, ".2e").ljust(10),
                                              format(r_primal[k - 1], ".2e").ljust(11),
                                              format(r_dual[k - 1], ".2e").ljust(9),
                                              format(time() - start, ".2e").ljust(8)))



    # Unscale and return x_i^(k+1/2).
    # if precond and has_constr:
    #     x_final = [ei * x for x, ei in zip(x_final, e_pre)]
    end = time()
    if verbose:
        print("----------------------------------------------------")
        if k < max_iter:
            print("Status: Solved")
        else:
            print("Status: Reach maximum iterations")
        print("Solve time: {:.2e}".format(end - start))
        print("Total number of iterations: {}".format(k))
        print("Best total residual: {:.2e}; reached at iteration {}".format(r_best, k_best))
        print(ddashes)
    return {"x_vals": x_final, "primal": np.array(r_primal[:k]), "dual": np.array(r_dual[:k]), \
            "num_iters": k, "solve_time": (end - start), "time_iter":time_iter[:k], "r_dr":r_dr[:k]}