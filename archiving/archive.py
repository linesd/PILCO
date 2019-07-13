import numpy as np

def archive_data(path, rollout, X, trajectories, cost_traj, uncer_trajs, uncer_costs):

    print("Archiving ...")

    name = "states_actions_"+str(rollout)+".npy"
    np.save(path+name, X, allow_pickle=True)

    name = "trajectories_" + str(rollout) + ".npy"
    np.save(path+name, trajectories, allow_pickle=True)

    name = "cost_trajectories_" + str(rollout) + ".npy"
    np.save(path+name, cost_traj, allow_pickle=True)

    name = "uncertainties_from_traj_" + str(rollout) + ".npy"
    np.save(path+name, uncer_trajs, allow_pickle=True)

    name = "uncertainties_from_costs_" + str(rollout) + ".npy"
    np.save(path+name, uncer_costs, allow_pickle=True)



