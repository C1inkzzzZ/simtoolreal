"""Widen the released SimToolReal checkpoint for extra observation dims, without changing what it computes.

The released networks read [obs (140) | SAPG extra params (32)] (actor LSTM) and [state (162) | extra params (32)]
(central-value MLP): rl_games appends a block id to the env obs, and the network swaps it for the block's learned
32-d vector (network_builder.py, 'extra_param'). New env dims are appended after the old ones, so in the first
weight matrix they land at column index old_size, just before the 32 extra-param columns. This script
  * inserts zero columns there in every weight matrix that reads [obs | params] or [state | params];
  * extends each input normalizer (RunningMeanStd) with mean 0, var 1 for the new dims.
With zero columns the pre-activations do not depend on the new inputs, so the widened policy and critic output
exactly what the released ones output, whatever the new dims contain (checked by --verify).

Only weights are kept (optimizer / rollout buffers are dropped): use checkpoint_load_mode=weights.

Usage:
  python finetune_velocity/expand_checkpoint.py --src pretrained_policy/model.pth \
      --dst pretrained_policy/model_velocity_obs149_state173.pth --new-obs 9 --new-state 11
"""

import argparse
from typing import Dict, Tuple

import torch

RESUME_ONLY_KEYS = (
    "optimizer", "obs", "rnn_states", "dones", "current_rewards", "current_shaped_rewards", "current_lengths",
    "env_state", "trackers", "scaler",
)


def _weights(ckpt):
    """rl_games saves {rank: weights}; older files are the weights dict itself."""
    if isinstance(ckpt, dict) and 0 in ckpt and isinstance(ckpt[0], dict):
        return ckpt[0], True
    return ckpt, False


def _normalizer_size(sd: Dict[str, torch.Tensor]) -> int:
    keys = [k for k in sd if k.endswith("running_mean_std.running_mean")]
    assert len(keys) == 1, f"expected one input normalizer, found {keys}"
    return int(sd[keys[0]].shape[0])


def _param_size(sd: Dict[str, torch.Tensor]) -> int:
    keys = [k for k in sd if k.endswith("extra_params")]
    assert len(keys) == 1, f"expected one SAPG extra_params tensor, found {keys}"
    return int(sd[keys[0]].shape[1])


def expand_state_dict(sd: Dict[str, torch.Tensor], n_new: int, name: str) -> Tuple[Dict[str, torch.Tensor], list]:
    """Return a widened copy of one network's state dict and a log of what changed."""
    old = _normalizer_size(sd)
    p = _param_size(sd)
    out, log = {}, []
    n_input_mats = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if k.endswith("running_mean_std.running_mean") and v.shape == (old,):
            out[k] = torch.cat([v, torch.zeros(n_new, dtype=v.dtype)])
            log.append(f"{name}: {k} {tuple(v.shape)} -> {tuple(out[k].shape)} (new mean 0)")
        elif k.endswith("running_mean_std.running_var") and v.shape == (old,):
            out[k] = torch.cat([v, torch.ones(n_new, dtype=v.dtype)])
            log.append(f"{name}: {k} {tuple(v.shape)} -> {tuple(out[k].shape)} (new var 1)")
        elif v.dim() == 2 and v.shape[1] == old + p:
            zeros = torch.zeros(v.shape[0], n_new, dtype=v.dtype)
            out[k] = torch.cat([v[:, :old], zeros, v[:, old:]], dim=1)
            n_input_mats += 1
            log.append(f"{name}: {k} {tuple(v.shape)} -> {tuple(out[k].shape)} (zero cols at {old}:{old + n_new})")
        else:
            out[k] = v
    assert n_input_mats >= 1, f"{name}: no weight matrix reads [input ({old}) | params ({p})]"
    return out, log


def expand_checkpoint(ckpt, n_new_obs: int, n_new_state: int):
    w, wrapped = _weights(ckpt)
    new_w, log = {}, []
    for k, v in w.items():
        if k in RESUME_ONLY_KEYS:
            log.append(f"dropped {k} (resume-only)")
            continue
        if k == "model":
            new_w[k], lg = expand_state_dict(v, n_new_obs, "actor")
            log += lg
        elif k == "assymetric_vf_nets":
            new_w[k], lg = expand_state_dict(v, n_new_state, "critic")
            log += lg
        elif k == "running_mean_std" and isinstance(v, dict):
            # Some saves also keep the actor normalizer at the top level (get_stats_weights(model_stats=True)).
            old = int(v["running_mean"].shape[0])
            new_w[k] = dict(v)
            new_w[k]["running_mean"] = torch.cat([v["running_mean"], torch.zeros(n_new_obs, dtype=v["running_mean"].dtype)])
            new_w[k]["running_var"] = torch.cat([v["running_var"], torch.ones(n_new_obs, dtype=v["running_var"].dtype)])
            log.append(f"top-level running_mean_std {old} -> {old + n_new_obs}")
        else:
            new_w[k] = v
    if "epoch" in new_w:
        new_w["epoch"] = 0
    if "frame" in new_w:
        new_w["frame"] = 0
    return ({0: new_w} if wrapped else new_w), log


def verify(src_w, dst_w, n_new_obs: int, n_new_state: int):
    """Check that kept entries are unchanged and the inserted columns are zero."""
    for net, n_new in (("model", n_new_obs), ("assymetric_vf_nets", n_new_state)):
        a, b = src_w[net], dst_w[net]
        old = _normalizer_size(a)
        for k, v in a.items():
            if not torch.is_tensor(v):
                continue
            u = b[k]
            if u.shape == v.shape:
                assert torch.equal(u, v), f"{net}.{k} changed"
            elif v.dim() == 1:
                assert torch.equal(u[:old], v), f"{net}.{k} old part changed"
            else:
                assert torch.equal(u[:, :old], v[:, :old]) and torch.equal(u[:, old + n_new:], v[:, old:]), k
                assert torch.count_nonzero(u[:, old:old + n_new]) == 0, f"{net}.{k} new columns not zero"
    print("verify: kept weights identical, new columns zero")


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="pretrained_policy/model.pth")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--new-obs", type=int, default=9)
    ap.add_argument("--new-state", type=int, default=11)
    args = ap.parse_args()
    ckpt = load(args.src)
    new_ckpt, log = expand_checkpoint(ckpt, args.new_obs, args.new_state)
    print("\n".join(log))
    verify(_weights(ckpt)[0], _weights(new_ckpt)[0], args.new_obs, args.new_state)
    torch.save(new_ckpt, args.dst)
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
