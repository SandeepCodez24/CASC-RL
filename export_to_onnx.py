
import torch
import torch.nn as nn
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from world_model.world_model import WorldModel
from agents.policy_network import ActorNetwork

def export_world_model():
    print("Exporting World Model...")
    wm = WorldModel(device="cpu")
    wm.load("world_model_best.pt", map_location="cpu")
    
    # Input: state (batch, 8), action (batch,). 
    # But for ONNX export, we need to wrap the internal network or handle the action input.
    # The WorldModel wrapper handles conversion to tensor internally.
    # Let's export the UNDERLYING network for simplicity in JS.
    # network(s_norm, a)
    dummy_s = torch.randn(1, 8)
    dummy_a = torch.zeros(1, dtype=torch.long)
    
    torch.onnx.export(
        wm.network,
        (dummy_s, dummy_a),
        "frontend/public/world_model.onnx",
        input_names=["state", "action"],
        output_names=["next_state_mean", "next_state_std"],
        dynamic_axes={"state": {0: "batch"}, "action": {0: "batch"}},
        opset_version=14
    )
    print("World Model Exported.")

def export_mappo():
    print("Exporting MAPPO Actor...")
    # ActorNetwork(state_dim=8, n_actions=5, predict_k=5)
    actor = ActorNetwork(state_dim=8, n_actions=5, predict_k=5)
    ckpt = torch.load("mappo_best.pt", map_location="cpu")
    
    # Handle multi-agent structure: ckpt['actors'] is a list of state_dicts
    if "actors" in ckpt:
        state_dict = ckpt["actors"][0]
    elif "network" in ckpt:
        state_dict = ckpt["network"]
    else:
        state_dict = ckpt
        
    actor.load_state_dict(state_dict)
    actor.eval()
    
    dummy_s_t = torch.randn(1, 8)
    dummy_s_future = torch.randn(1, 5, 8)
    
    torch.onnx.export(
        actor,
        (dummy_s_t, dummy_s_future),
        "frontend/public/mappo.onnx",
        input_names=["s_t", "s_future"],
        output_names=["logits"],
        dynamic_axes={"s_t": {0: "batch"}, "s_future": {0: "batch"}},
        opset_version=14
    )
    print("MAPPO Exported.")

if __name__ == "__main__":
    os.makedirs("frontend/public", exist_ok=True)
    export_world_model()
    export_mappo()
