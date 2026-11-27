"""Continue training from the released SimToolReal checkpoint, using the config it was trained with.

Loads pretrained_policy/config.yaml (the checkpoint's own full config), applies a few overrides for a single-GPU
cluster run, and starts rl_games training. Only the weights are loaded by default (fresh optimizer and learning-rate
schedule, epoch 0), which is what fine-tuning needs.

--task static    the official task (sanity runs; what 00_ft_sanity used).
--task velocity    SimToolRealVelocity: pose + velocity goals (finetune_velocity/velocity.yaml, README sections 3-13).
                   Adds the goal_velocity obs (policy + critic) and velocity_phase (critic).
--task trajectory  SimToolRealTrajectory: smooth trajectory goals (finetune_velocity/trajectory.yaml, README section 14).
                   Additionally goal_lookahead (policy + critic) and traj_progress (critic).
Both widen the checkpoint with zero weights for whatever obs terms it lacks (expand_checkpoint.py; cached next to the
source checkpoint, so it works from the released model or from a velocity fine-tune), and by default train only the
critic for the first --critic-warmup epochs.

SAPG: the checkpoint has one learned row per exploration block (extra_params and sigma), 6 in total, so the run keeps
6 blocks and only the block size (num_envs / 6) changes.

Examples:
  PYTHONPATH=$PWD python finetune_velocity/launch_from_pretrained.py --num-envs 6144 --name 00_ft_sanity --max-epochs 300
  PYTHONPATH=$PWD python finetune_velocity/launch_from_pretrained.py --task velocity --num-envs 12288 --name 01_vel
  extra overrides: ... task.env.velocity.velocityGoalProb=0.3 train.params.config.learning_rate=3e-5
"""
# isort: off
from isaacgym import gymapi  # noqa: F401  (Isaac Gym must be imported before torch)
# isort: on
import argparse
import os
import sys

import torch
import yaml
from omegaconf import OmegaConf

import isaacgymenvs  # noqa: F401  (registers the OmegaConf resolvers used in the saved config)
from deployment.isaac.isaac_env import merge_cfg_with_default_config
from deployment.rl_player_utils import read_cfg_omegaconf
from isaacgymenvs.train import launch_rlg_hydra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expand_checkpoint  # noqa: E402

NUM_SAPG_BLOCKS = 6  # fixed by the checkpoint
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_OBS, BASE_STATE = 140, 162  # released obsList / stateList
EXTRA_SIZES = {"goal_velocity": 9, "velocity_phase": 2, "goal_lookahead": 15, "traj_progress": 1}
# extra terms appended (in this order) to the released obsList / stateList
TASKS = {
    "velocity": dict(name="SimToolRealVelocity", obs=["goal_velocity"],
                     state=["goal_velocity", "velocity_phase"]),
    "trajectory": dict(name="SimToolRealTrajectory", obs=["goal_velocity", "goal_lookahead"],
                       state=["goal_velocity", "velocity_phase", "goal_lookahead", "traj_progress"]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="static", choices=["static", "velocity", "trajectory"])
    p.add_argument("--config", default="pretrained_policy/config.yaml")
    p.add_argument("--checkpoint", default="pretrained_policy/model.pth",
                   help="released checkpoint (velocity: widened automatically, or pass an already widened one)")
    p.add_argument("--num-envs", type=int, default=6144)
    p.add_argument("--name", default="00_ft_sanity", help="run name; must start with digits and '_' (e.g. 00_)")
    p.add_argument("--max-epochs", type=int, default=300, help="PPO iterations; -1 runs until the job ends")
    p.add_argument("--load-mode", default="weights", choices=["weights", "resume"],
                   help="weights: fresh optimizer, epoch 0 (fine-tuning). resume: also optimizer, epoch, env state")
    p.add_argument("--lr", type=float, default=None, help="actor learning rate (velocity default 5e-5)")
    p.add_argument("--critic-warmup", type=int, default=None,
                   help="epochs with actor lr 0 (critic only) at the start (velocity default 20, static 0)")
    p.add_argument("--save-every", type=int, default=200, help="epochs between saved checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*", help="extra key=value config overrides (values parsed as YAML)")
    return p.parse_args()


def widened_checkpoint(src: str, target_obs: int, target_state: int) -> str:
    """Return a checkpoint whose input layers fit target_obs / target_state; create it once next to src.

    The new terms are always appended after the ones the checkpoint already reads (TASKS lists them in that order),
    so the zero columns go at the end of the old inputs, before the SAPG extra-param columns.
    """
    ckpt = expand_checkpoint.load(src)
    w = expand_checkpoint._weights(ckpt)[0]
    obs_size = expand_checkpoint._normalizer_size(w["model"])
    state_size = expand_checkpoint._normalizer_size(w["assymetric_vf_nets"])
    n_obs, n_state = target_obs - obs_size, target_state - state_size
    assert n_obs >= 0 and n_state >= 0, f"{src} reads more inputs ({obs_size}/{state_size}) than the task provides"
    if n_obs == 0 and n_state == 0:
        print(f"[launch] {src}: inputs {obs_size}/{state_size} already match")
        return src
    dst = os.path.splitext(src)[0] + f"_obs{target_obs}_state{target_state}.pth"
    if not os.path.exists(dst):
        new_ckpt, log = expand_checkpoint.expand_checkpoint(ckpt, n_obs, n_state)
        print("\n".join(log))
        expand_checkpoint.verify(w, expand_checkpoint._weights(new_ckpt)[0], n_obs, n_state)
        torch.save(new_ckpt, dst)
        print(f"[launch] widened checkpoint ({obs_size}/{state_size} -> {target_obs}/{target_state}) saved to {dst}")
    else:
        print(f"[launch] using cached widened checkpoint {dst}")
    return dst


def configure_task(cfg, task: str):
    """Velocity / trajectory task on top of the released config. Returns (target obs size, target state size)."""
    t = TASKS[task]
    env = cfg.task.env
    env.velocity = OmegaConf.load(os.path.join(HERE, "velocity.yaml"))
    if task == "trajectory":
        env.trajectory = OmegaConf.load(os.path.join(HERE, "trajectory.yaml"))
    cfg.task.name = t["name"]
    cfg.task_name = t["name"]
    obs = list(env.obsList)
    state = list(env.stateList)
    assert len(obs) == 10 and len(state) == 18, f"unexpected released obs/state lists: {obs} / {state}"
    env.obsList = obs + t["obs"]
    env.stateList = state + t["state"]
    # The released config starts the tolerance curriculum at 0.075; the checkpoint was trained down to 0.01 but that
    # value lives in the env state, which weights-only loading does not restore.
    env.successTolerance = 0.01
    return (BASE_OBS + sum(EXTRA_SIZES[k] for k in t["obs"]), BASE_STATE + sum(EXTRA_SIZES[k] for k in t["state"]))


def install_load_and_warmup_patches(load_mode: str, warmup_epochs: int, base_lr: float):
    from rl_games import torch_runner
    from rl_games.algos_torch import torch_ext
    from rl_games.common import a2c_common

    # 1) checkpoint_load_mode, and a strict central-value load: rl_games' weights mode silently skips a critic whose
    #    shapes do not match, which would hide a wrong widening.
    def strict_restore(agent, run_args):
        ckpt = run_args.get("checkpoint")
        if not ckpt:
            return
        if load_mode == "resume":
            agent.restore(ckpt)
            return
        weights = torch_runner._load_checkpoint_weights(agent, ckpt)
        agent.set_weights(weights)
        if getattr(agent, "has_central_value", False):
            assert "assymetric_vf_nets" in weights, "checkpoint has no central value weights"
            agent.central_value_net.load_state_dict(weights["assymetric_vf_nets"])  # raises on mismatch
        print(f"=> initialized actor and critic weights from '{ckpt}' (strict)")

    torch_runner._restore = strict_restore
    _ = torch_ext  # (kept for clarity: rl_games loads with torch_ext.load_checkpoint inside _load_checkpoint_weights)

    # 2) critic warmup: actor lr 0 for the first epochs. The adaptive schedule raises the lr while the KL is 0, so the
    #    lr it computes is reset to base_lr after every warmup epoch.
    if warmup_epochs > 0:
        cls = a2c_common.ContinuousA2CBase
        orig_train_epoch = cls.train_epoch
        orig_update_lr = a2c_common.A2CBase.update_lr

        def update_lr(self, lr):
            if getattr(self, "_ft_warmup", False):
                lr = 0.0
            return orig_update_lr(self, lr)

        def train_epoch(self):
            self._ft_warmup = self.epoch_num <= warmup_epochs
            if self._ft_warmup:
                for g in self.optimizer.param_groups:
                    g["lr"] = 0.0
                if self.epoch_num in (1, warmup_epochs):
                    print(f"[launch] critic warmup epoch {self.epoch_num}/{warmup_epochs}: actor lr 0")
            out = orig_train_epoch(self)
            if self._ft_warmup:
                self.last_lr = base_lr
                self._ft_warmup = False
                if self.epoch_num == warmup_epochs:
                    orig_update_lr(self, base_lr)
                    print(f"[launch] critic warmup done, actor lr {base_lr}")
            return out

        a2c_common.A2CBase.update_lr = update_lr
        cls.train_epoch = train_epoch


def main():
    args = parse_args()
    assert args.num_envs % NUM_SAPG_BLOCKS == 0, f"--num-envs must be a multiple of {NUM_SAPG_BLOCKS}"
    assert args.name.split("_")[0].isdigit(), "--name must start with digits and '_' (e.g. 00_ft)"
    velocity = args.task in ("velocity", "trajectory")

    cfg = read_cfg_omegaconf(config_path=args.config, device="cuda:0")
    cfg = merge_cfg_with_default_config(cfg)
    OmegaConf.set_struct(cfg, False)

    n = args.num_envs
    tc = cfg.train.params.config
    cfg.task.env.numEnvs = n
    cfg.num_envs = n
    tc.num_actors = n
    tc.expl_coef_block_size = n // NUM_SAPG_BLOCKS
    batch = n * int(tc.horizon_length)
    for c in (tc, tc.get("central_value_config")):
        if c is not None and (int(c.minibatch_size) > batch or batch % int(c.minibatch_size) != 0):
            c.minibatch_size = batch
    tc.name = args.name
    cfg.full_experiment_name = args.name  # rl_games run folder: runs/<name>/ (nn/ checkpoints, summaries/)
    cfg.experiment = args.name
    tc.max_epochs = args.max_epochs
    tc.save_frequency = args.save_every

    lr = args.lr if args.lr is not None else (5e-5 if velocity else None)
    if lr is not None:
        tc.learning_rate = lr
    warmup = args.critic_warmup if args.critic_warmup is not None else (20 if velocity else 0)

    ckpt = os.path.abspath(args.checkpoint)
    if velocity:
        target_obs, target_state = configure_task(cfg, args.task)
        if args.load_mode == "weights":
            ckpt = os.path.abspath(widened_checkpoint(ckpt, target_obs, target_state))
    cfg.checkpoint = ckpt
    cfg.train.params.load_checkpoint = True
    cfg.train.params.load_path = ckpt

    # Cluster compute nodes have no graphics: no cameras, no rendering, no videos.
    cfg.headless = True
    cfg.test = False
    cfg.capture_video = False
    cfg.force_render = False
    cfg.task.env.enableCameraSensors = False
    cfg.task.env.capture_video = False
    cfg.task.env.capture_viewer = False
    cfg.wandb_activate = False
    cfg.pbt.enabled = False
    cfg.multi_gpu = False
    cfg.sim_device = cfg.rl_device = "cuda:0"
    cfg.graphics_device_id = 0
    cfg.seed = args.seed

    for item in args.overrides:
        key, value = item.split("=", 1)
        OmegaConf.update(cfg, key, yaml.safe_load(value), force_add=False)
        print(f"override {key} = {OmegaConf.select(cfg, key)}")

    print(f"task {cfg.task_name}, num_envs {n}, SAPG {NUM_SAPG_BLOCKS} blocks x {tc.expl_coef_block_size}, "
          f"batch {batch}, minibatch {tc.minibatch_size}, lr {tc.learning_rate}, critic warmup {warmup}, "
          f"load mode {args.load_mode}, successTolerance {cfg.task.env.successTolerance}, ckpt {ckpt}")
    if velocity:
        print(f"obsList {list(cfg.task.env.obsList)}\nstateList {list(cfg.task.env.stateList)}")
        print("velocity cfg:\n" + OmegaConf.to_yaml(cfg.task.env.velocity))
        if args.task == "trajectory":
            print("trajectory cfg:\n" + OmegaConf.to_yaml(cfg.task.env.trajectory))

    install_load_and_warmup_patches(args.load_mode, warmup, float(tc.learning_rate))
    launch_rlg_hydra(cfg)


if __name__ == "__main__":
    main()
