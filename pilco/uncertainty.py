import numpy as np
from pilco.models.mbr import BR
from math import floor
from examples.utils import policy

class Uncertainty():

    def __init__(self, pilco, env, low, high):
        print("Uncertainty initiated ... ")
        self.pilco=pilco
        self.env=env
        self.data_max = high
        self.data_min = low

    def create_models(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]
        self.models = []

        for i in range(self.num_outputs):
            self.models.append(BR())
            self.models[i].add_data(X, Y[:, i:i+1])

    def optimise_models(self, maxiter=200, disp=False):
        for i in range(self.num_outputs):
            self.models[i].optimise(maxiter=maxiter, disp=disp)

    def mc_rollout(self, M, N, T):
        """
        :param M: number of sets of weights sampled
        :param N: number of trajectories
        :param T: number of time steps for each trajectory
        :return:
        """
        print("Running Monte Carlo rollouts ...")
        self.M=M
        self.N=N
        self.T=T
        trajectories = []
        trajectories_actions = []
        for m in range(self.M):
            traj_m=[]
            traj_a_m=[]
            W = self.get_weights()
            states = self.get_n_starts(self.N)
            traj_m.append(states.copy())
            for t in range(1, self.T):
                states_actions = self.augment_actions(states.copy(),is_random=False)
                traj_a_m.append(states_actions.copy())
                delta_states = self.cascade(states_actions, W)
                next_states = self.get_next_state(states, delta_states)
                traj_m.append(next_states.copy())
                states = next_states
            trajectories.append(traj_m)
            trajectories_actions.append(traj_a_m)
            if m % 10 == 0:
                print("Completed %i of %i MC rollouts" %(m*t*N, M*N*T))

        return trajectories, trajectories_actions

    def get_weights(self):
        W = []
        for i in range(self.num_outputs):
            W.append(self.models[i].sample_weights(1)) # Draw weights for each model
        return W

    def get_n_starts(self, N):
        state = []
        # theta = np.random.normal(np.pi,1,size=(N, 1))
        # state.append(np.cos(theta))
        # state.append(np.sin(theta))
        # state.append(np.random.uniform(-1,1,size=(N, 1)))
        theta = np.random.normal(0, 0.2, size=(N, 1))
        for i in range(len(theta)):
            if theta[i] >= 0:
                theta[i] -= np.pi
            else:
                theta[i] += np.pi

        state.append(np.cos(theta))
        state.append(np.sin(theta))
        state.append(np.random.uniform(-0.5,0.5,size=(N, 1)))
        # state.append(-1*np.ones((N, 1)))
        # state.append(np.zeros((N, 1)))
        # state.append(np.zeros((N, 1)))
        # theta = np.random.uniform(0, 2*np.pi, size=(N, 1))
        # state.append(np.cos(theta))
        # state.append(np.sin(theta))
        # state.append(np.random.uniform(self.data_min[2], self.data_max[2], size=(N, 1)))
        # for i in range(self.num_outputs):
        #     state.append(np.random.uniform(self.data_min[i], self.data_max[i], size=(N, 1)))
        return state

    def augment_actions(self, states, is_random):
        N = states[0].shape[0]
        if is_random:
            states.append(np.random.uniform(self.data_min[-1], self.data_max[-1], size=(N, 1)))
        else:
            actions=np.zeros((N,))
            for i in range(N):
                state = np.array([states[0][i], states[1][i], states[2][i]]).squeeze()
                actions[i] = policy(self.env,self.pilco, state, random=is_random)
            states.append(actions.reshape(-1,1))
        return states

    def cascade(self, state, W):
        next_state=[]
        for i in range(self.num_outputs):
            next_state.append(self.models[i].post_one_step(np.array(state).squeeze().T, W[i]))
        return next_state

    def predict(self, state):
        predict=[]
        for i in range(self.num_outputs):
            predict.append(self.models[i].pred_one_step(state))
        return predict

    def get_next_state(self, states, delta_states):
        next_states=[]
        for i in range(self.num_outputs):
            next_states.append(states[i] + delta_states[i])
        return next_states

    def disentangle_trajectories(self, trajectories):
        traj = np.array(trajectories).squeeze()
        total = traj.var()
        epistemic = traj.mean(axis=(1,2,3)).var()
        aleatoric = traj.var(axis=(1,2,3)).mean()

        return total, aleatoric, epistemic

    def disentangle_costs(self, cost):
        traj = np.array(cost).squeeze()
        total = traj.var()
        epistemic = traj.mean(axis=(1,2)).var()
        aleatoric = traj.var(axis=(1,2)).mean()

        return total, aleatoric, epistemic

    def check_model_mse(self, X, Y):
        ind = int(floor(0.8 * len(X[0])))
        X_train = X[:ind, :]
        Y_train = Y[:ind, :]
        X_test = X[ind:, :]
        Y_test = Y[ind:, :]
        self.create_models(X_train, Y_train)
        self.optimise_models()
        Y_predict = self.predict(X_test)
        SE=0
        for i in range(self.num_outputs):
            SE+=((Y_predict[i][0].squeeze()-Y_test[:,i])**2).mean()
        return SE.mean()

    def compute_cost(self, trajectories, target, s):
        ave_cost=0
        m_costs=[]
        for m in range(self.M):
            t_costs = []
            for t in range(self.T):
                n_costs = np.zeros((self.N, 1))
                for n in range(self.N):
                    x = np.array([trajectories[m][t][0][n], trajectories[m][t][1][n], trajectories[m][t][2][n]]).reshape(-1, 1)
                    sqdist = ((x - target) ** 2).sum()
                    cost = 1 - np.exp(-(1 / (2 * s ** 2) * sqdist))
                    ave_cost += cost
                    n_costs[n] = cost
                t_costs.append(n_costs.copy())
            m_costs.append(t_costs.copy())
        ave_cost /= (self.M * self.N * self.T)
        return ave_cost, m_costs

