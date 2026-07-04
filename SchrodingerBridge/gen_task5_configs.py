import json
import copy

with open("exp/task5_baseline_2ep/config.json", "r") as f:
    base_cfg = json.load(f)

configs = {
    "task5_endpoint_a_2ep": {
        "training_objective_mode": "endpoint",
        "w_endpoint_content": 1.0,
        "w_endpoint_style": 8.0,
        "w_endpoint_velocity_reg": 0.0,
        "notes": "Task 5 Endpoint A: endpoint mode (content=1.0, style=8.0, vel_reg=0.0) with FiLM + fixed_one gate + no GN - 2 epochs"
    },
    "task5_endpoint_b_2ep": {
        "training_objective_mode": "endpoint",
        "w_endpoint_content": 1.0,
        "w_endpoint_style": 8.0,
        "w_endpoint_velocity_reg": 0.1,
        "notes": "Task 5 Endpoint B: endpoint mode (content=1.0, style=8.0, vel_reg=0.1) with FiLM + fixed_one gate + no GN - 2 epochs"
    },
    "task5_endpoint_c_2ep": {
        "training_objective_mode": "endpoint",
        "w_endpoint_content": 0.5,
        "w_endpoint_style": 16.0,
        "w_endpoint_velocity_reg": 0.05,
        "notes": "Task 5 Endpoint C: endpoint mode (content=0.5, style=16.0, vel_reg=0.05) with FiLM + fixed_one gate + no GN - 2 epochs"
    }
}

for exp_name, params in configs.items():
    cfg = copy.deepcopy(base_cfg)
    
    cfg["bridge"]["training_objective_mode"] = params["training_objective_mode"]
    cfg["bridge"]["w_endpoint_content"] = params["w_endpoint_content"]
    cfg["bridge"]["w_endpoint_style"] = params["w_endpoint_style"]
    cfg["bridge"]["w_endpoint_velocity_reg"] = params["w_endpoint_velocity_reg"]
    
    cfg["checkpoint"]["save_dir"] = f"./exp/{exp_name}"
    cfg["ablation"]["name"] = exp_name
    cfg["ablation"]["notes"] = params["notes"]
    
    import os
    os.makedirs(f"exp/{exp_name}", exist_ok=True)
    
    with open(f"exp/{exp_name}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Created config for {exp_name}")

print("All configs created successfully!")
