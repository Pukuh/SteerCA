import torch
import tqdm
from tensordict import TensorDict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import steerca.core.cs_env as cs_env
import steerca.firebreak.agents as ff_agents

agent_params = TensorDict(
    {
    "params": TensorDict(
        {
            "tc": 1,  # time to cut a tree
            "tm": 2,  # time to move
            "n": 10,  # size of the grid
        }, 
        [],
    ),
    },
    [],      
)
agent_params = agent_params.expand([1])  # Expand to batch size of 1


#define a main function to create and test the environment

def main():
    # Create a grid of size 10x10 with random values between 0 and 2
    grid = torch.ones((1, 10, 10), dtype=torch.int64)
    grid[0,0,0] = 2  # Set the first cell to FIRE

    # Create a tensordict with the grid and agent
    tensordict = TensorDict(
        {
            "grid": grid,
            "t": torch.tensor([1.0]),
            "action": torch.tensor([0]),  
            "agent_state": torch.tensor([[8,8]]),# initial action
            "agent_params": agent_params,
        },
        batch_size=[1]  # Batch size of 1 for simplicity
    )

    funcs = {
        "fun_ca_step": ff_agents.ca_step,
        "fun_time": ff_agents.time,
        "fun_action": ff_agents.action,
        "fun_reward": ff_agents.reward,
    }

    # Step through the environment
    for _ in tqdm.tqdm(range(3)):
        tensordict = cs_env.step(tensordict, funcs=funcs)
        tensordict["action"] = torch.randint(0, 9, (1,))  # Random action
        print(tensordict["grid"])

if __name__ == "__main__":
    main()

