import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym, d4rl
import d4rl.gym_mujoco
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))
# Make diffusion_policy importable without moving files
import sys, os
sys.path.append(os.path.join(base_path, "..", "diffusion_policy"))

	


@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):

	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	OmegaConf.resolve(cfg)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	# cfg.use_wandb = False

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=OmegaConf.to_container(cfg, resolve=True),
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

	num_env = cfg.env.n_envs
	def make_env():
		if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			env = make_robomimic_env(env=cfg.env_name, 
							normalization_path=cfg.normalization_path, 
							low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, 
							dppo_path=cfg.dppo_path, 
							abs_action=cfg.env.wrappers.robomimic_lowdim.get("abs_action", False),)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	base_policy = load_base_policy(cfg)

	# determine diffusion action dims (T, D)
	if getattr(cfg, "policy", None) is not None:
		act_steps = getattr(cfg.policy, "n_action_steps", None) or getattr(cfg.policy, "horizon", None)
		# action_dim = getattr(cfg.policy, "action_dim", None) or getattr(cfg, "action_dim", None)
		action_dim = getattr(cfg, "action_dim", None)
	else:
		act_steps = getattr(cfg, "act_steps", None)
		action_dim = getattr(cfg, "action_dim", None)

	diffusion_act_dim = (int(act_steps), int(action_dim))

	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(cfg.seed + 1)
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=0.0,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)
	if cfg.algorithm == 'dsrl_sac':
		model = SAC(
			"MlpPolicy",
			env,
			learning_rate=cfg.train.actor_lr,
			buffer_size=20000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
		)
	elif cfg.algorithm == 'dsrl_na':
		checkpoint_path = getattr(cfg, "checkpoint_path", None)
		
		if checkpoint_path and os.path.exists(checkpoint_path):
			print(f"Loading pre-trained DSRL model from: {checkpoint_path}")
            # Use the .load() method. Note that we pass the new env and 
            # any updated parameters (like diffusion_policy) manually.
			model = DSRL.load(
                checkpoint_path,
                env=env,
				custom_objects={
					"diffusion_policy": base_policy,
					# Force these specific attributes to these specific values
					"diffusion_act_chunk": diffusion_act_dim[0], # e.g., 8
					"diffusion_act_dim":   diffusion_act_dim[1], # e.g., 7
				},
                device=cfg.device, # Ensure it loads to the right device
            )
		else:
			print("Initializing new DSRL model.")
			model = DSRL(
                "MlpPolicy",
                env,
                # ... (rest of your existing initialization arguments)
                diffusion_policy=base_policy,
                diffusion_act_dim=diffusion_act_dim,
                noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
                critic_backup_combine_type=cfg.train.critic_backup_combine_type,
            )

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval, 
		save_path=cfg.logdir+'/checkpoint/',
		name_prefix='ft_policy',
		save_replay_buffer=cfg.save_replay_buffer, 
		save_vecnormalize=True,
	)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
		print("Using DiffusionPolicyEnvWrapper")
	eval_env.seed(cfg.seed + num_env + 1) 

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = int(cfg.num_evals / num_env_eval), 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=eval_env, 
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm=cfg.algorithm,
		max_steps=MAX_STEPS,
		deterministic_eval=cfg.deterministic_eval,
	)
	logging_callback.evaluate(model, deterministic=False)
	if cfg.deterministic_eval:
		logging_callback.evaluate(model, deterministic=True)
	logging_callback.log_count += 1

	# Close environment and wandb
	env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
