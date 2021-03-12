# Author: Kaituo Xu, Fan Yu

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    alphas = [[0 for _ in range(T)] for _ in range(N)]
    for i in range(N):
        alphas[i][0] = pi[i] * B[i][O[0]]
    for j in range(1, T):
        for i in range(N):
            alphas[i][j] = sum(alphas[k][j - 1] * A[k][i] * B[i][O[j]] for k in range(N))
    prob = sum(e[-1] for e in alphas)

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    betas = [[1 for _ in range(T)] for _ in range(N)]
    for j in range(T - 2, -1, -1):
        for i in range(N):
            betas[i][j] = sum(betas[k][j + 1] * A[i][k] * B[k][O[j + 1]] for k in range(N))
    prob = sum(betas[i][0] * pi[i] * B[i][O[0]] for i in range(N))

    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    delta = [[0 for _ in range(T)] for _ in range(N)]
    phi = [[0 for _ in range(T)] for _ in range(N)]
    for i in range(N):
        delta[i][0] = pi[i] * B[i][O[0]]
    for j in range(1, T):
        for i in range(N):
            max_prob = 0.0
            state = 0
            for k in range(N):
                prob = delta[k][j - 1] * A[k][i] * B[i][O[j]]
                if prob > max_prob:
                    max_prob = prob
                    state = k
            delta[i][j] = max_prob
            phi[i][j] = state
    Q = [1, 2, 3]
    state_index = 0
    for i in range(N):
        if delta[i][T - 1] > best_prob:
            best_prob = delta[i][T - 1]
            state_index = i
    best_path.append(Q[state_index])
    for j in range(T - 1, 0, -1):
        state_index = phi[state_index][j]
        best_path.append(Q[state_index])
    best_path.reverse()

    # End Assignment
    return best_prob, best_path


def _check_forward_and_backword_algorithms(O, HMM_model):
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)

    # alphas
    alphas = [[0 for _ in range(T)] for _ in range(N)]
    for i in range(N):
        alphas[i][0] = pi[i] * B[i][O[0]]
    for j in range(1, T):
        for i in range(N):
            alphas[i][j] = sum(alphas[k][j - 1] * A[k][i] * B[i][O[j]] for k in range(N))
    # betas
    betas = [[1 for _ in range(T)] for _ in range(N)]
    for j in range(T - 2, -1, -1):
        for i in range(N):
            betas[i][j] = sum(betas[k][j + 1] * A[i][k] * B[k][O[j + 1]] for k in range(N))

    probs = []
    for j in range(T):
        probs.append(sum(alphas[i][j] * betas[i][j] for i in range(N)))
    print(probs)
    for k in range(1, T):
        if probs[k] != probs[k - 1]:
            print("find wrong answer:", k - 1, k)


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)

    # _check_forward_and_backword_algorithms(observations, HMM_model)
