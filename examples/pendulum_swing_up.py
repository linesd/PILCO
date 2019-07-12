import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
from utils import rollout, policy
np.random.seed(0)
from pilco.uncertainty import Uncertainty
import matplotlib.pyplot as plt

# NEEDS a different initialisation than the one in gym (change the reset() method),
# to (m_init, S_init), modifying the gym env

# Introduces subsampling with the parameter SUBS and modified rollout function
# Introduces priors for better conditioning of the GP model
# Uses restarts

class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01*high) # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()


SUBS=3
bf = 30
maxiter=50
max_action=2.0
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])
m_init = np.reshape([-1.0, 0, 0.0], (1,3))
S_init = np.diag([0.01, 0.05, 0.01])
T = 40
T_sim = T
J = 4
N = 12
restarts = 2

with tf.Session() as sess:
    env = myPendulum()
    low = env.observation_space.low
    high = env.observation_space.high

    # Initial random rollouts to generate a dataset
    X,Y = rollout(env, None, timesteps=T, verbose=False, random=True, SUBS=SUBS)
    for i in range(1,J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # uncertainty
    uncertainty = Uncertainty(pilco, env, low, high)
    uncertainties_costs=np.zeros((N, 5))
    uncertainties_trajectories = np.zeros((N, 5))

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rol in range(N):
        print("**** ITERATION no", rol, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        MSE = uncertainty.check_model_mse(X, Y)
        print("MSE: ", MSE)
        uncertainty.create_models(X, Y)
        uncertainty.optimise_models()
        trajectories= uncertainty.mc_rollout(M=50, N=50, T=40)
        ave_cost, traj_costs = uncertainty.compute_cost(trajectories, target.reshape(-1, 1), 1)
        total_t, aleatoric_t, epistemic_t = uncertainty.disentangle_trajectories(trajectories)
        total_c, aleatoric_c, epistemic_c = uncertainty.disentangle_costs(traj_costs)
        uncertainties_costs[rol, :] = [rol, total_c, aleatoric_c, epistemic_c, ave_cost]
        uncertainties_trajectories[rol, :] = [rol, total_t, aleatoric_t, epistemic_t, ave_cost]
        print("Trajectories: Iter: %i  Total: %.6f  Epistemic: %.6f  Aleatoric: %.6f  Cost1: %.6f" %
              (rol, total_t, epistemic_t, aleatoric_t, ave_cost))
        print("Costs: Iter: %i  Total: %.6f  Epistemic: %.6f  Aleatoric: %.6f  Cost1: %.6f" %
              (rol, total_c, epistemic_c, aleatoric_c, ave_cost))
        np.savetxt("states_actions.csv", X, delimiter=",")
        np.savetxt("uncertainties_costs.csv", uncertainties_costs, delimiter=",")
        np.savetxt("uncertainties_trajectories.csv", uncertainties_trajectories, delimiter=",")

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        # cur_rew = 0
        # for t in range(0,len(X_new)):
        #     cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        # print('On this episode reward was ', cur_rew)

        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)
