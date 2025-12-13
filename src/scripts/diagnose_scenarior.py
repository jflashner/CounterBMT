"""
Diagnostic script to understand ScenarioEnv termination behavior.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

DATA_DIR = "./src/exp_converted"

# Create environment with minimal config (like the tutorial)
env = ScenarioEnv({
    "manual_control": False,
    "reactive_traffic": False,
    "use_render": False,
    "agent_policy": ReplayEgoCarPolicy,
    "data_directory": DATA_DIR,
    "num_scenarios": 10,
})

print(f"\n{'='*60}")
print("ScenarioEnv Diagnostic")
print(f"{'='*60}")

# Reset to scenario 0
obs, info = env.reset(seed=1)

print(f"\nAfter reset:")
print(f"  Info keys: {info.keys()}")
print(f"  Scenario ID: {env.engine.data_manager.current_scenario_id}")

# Run a few steps and check termination
for step in range(20):
    obs, reward, terminated, truncated, info = env.step([1.0, 0])
    
    print(f"\nStep {step}:")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")
    print(f"  reward: {reward:.4f}")
    
    # Print relevant info
    for key in ['crash_vehicle', 'crash_object', 'crash_human', 'out_of_road', 'arrive_dest']:
        if key in info:
            print(f"  {key}: {info[key]}")
    
    if terminated or truncated:
        print(f"\n*** TERMINATED at step {step} ***")
        print(f"Full info: {info}")
        break

env.close()
print(f"\n{'='*60}")