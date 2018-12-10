import numpy as np
#import cvxpy as cp
from scipy.optimize import minimize

def learn_representations(task_examples, task_labels, gamma=1.0, epsilon=0.001):
    d = len(task_examples[0][0]); T = len(task_examples)
    D = np.identity(d)/d; W = np.zeros((d,T))
    difference = 1000
    while difference > epsilon:
        old_W = np.array(W)
        for i in range(T):
            print(i)
            #initial = np.ndarray((d,)); print(initial)
            res = minimize(objective_func, np.zeros((d,)), args=(D, task_examples[i], task_labels[i], gamma), method='BFGS', options={'disp':True, 'eps':0.5}).x#, constraints={'type':'eq', 'fun': constraint_func, 'args':(D,)})
            # Fixing res to fit the constraint
            # while not constraint_func(res, D):
            #     pass
            W[:,[i]] = np.reshape(res, (len(res),1))
        # Update D
        # lamb_da = list(); sum = 0
        # for i in range(d):
        #     lamb_da_ = np.linalg.norm(W[i], ord=2)
        #     lamb_da.append(lamb_da_); sum += lamb_da_
        # lamb_da = np.array(lamb_da)/sum
        # D = np.resize(np.diag(lamb_da),(d,d))
        C = np.sqrt(np.dot(W,W.T))
        D = C/np.trace(C)
        difference = np.sum(np.abs(W - old_W))
        print(difference)
    return W

def objective_func(x, *args):
    D = args[0]; examples = args[1]; labels = args[2]; gamma = args[3]
    #print(D.shape); print(examples.shape); print(labels.shape)
    loss = np.sum((labels - np.dot(examples,x))**2)
    regulization = gamma*np.inner(x,np.dot(np.linalg.pinv(D),x))
    return loss + regulization

def constraint_func(b, D):
    #return minimize(cont_obj_func, np.ones(b.shape), args=(b,args[0]), method='BFGS').fun #constraint=[{'type':'ineq', 'fun':pos_cont},{'type':'ineq', 'fun':neg_cont}])
    try:
        np.linalg.solve(D,b)
    except np.linalg.LinAlgError:
        return -1
    else:
        return 0

def cont_obj_func(x, *args):
    return np.sum(np.abs(np.dot(args[1],x) - args[0]))

# def pos_cont(x):
#     return x
#
# def neg_cont(x):
#     return -x

# def learn_representations(task_examples, task_labels, gamma=1.0, epsilon=0.001):
#     d = len(task_examples[0][0]); T = len(task_examples)
#     D = np.identity(d); W = np.zeros((d,T))
#     difference = 1000
#     while(difference > epsilon):
#         print(difference)
#         old_W = np.array(W)
#         for i in range(T):
#             w = cp.Variable((d,1))
#             task_labels_ = cp.Parameter((len(task_labels[i]),1), value=np.reshape(task_labels[i],(len(task_labels[i]),1)))
#             task_examples_ = cp.Parameter(task_examples[i].shape, value=task_examples[i])
#             D_ = cp.Parameter((d,d), value=np.linalg.pinv(D))
#             loss = cp.sum(cp.abs(task_labels_ - task_examples_*w))  # CHECK THIS AGAIN
#             regulization = gamma*w.T*(D_*w)
#             problem = cp.Problem(cp.Minimize(loss + regulization))#,[cp.Pnorm(D*is_within_range(w,D)) == 0])
#             problem.solve()
#             W[i] = w.value
#         C = np.sqrt(np.dot(W,W.T))
#         D = C/np.trace(C)
#         difference = np.sum(np.abs(W - old_W))
#     return W
#
def run_evaluation(W, task_id, examples, labels):
    predicted_labels = list()
    for i in range(len(examples)):
        ft = np.inner(W[:,[task_id]].ravel(),examples[i])
        print(ft, labels[i])

def get_folds(examples, num_fold):
    pass
