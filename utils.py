import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import os
from typing import Union
from scipy.spatial.transform import Rotation

# Lightweight RotationTransformer without pytorch3d dependency
class RotationTransformer:
    """
    Minimal rotation transformer: converts between axis-angle (3D) and rotation_6d (6D).
    Uses scipy.spatial.transform.Rotation for the heavy lifting.
    """
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        assert from_rep in ['axis_angle', 'rotation_6d']
        assert to_rep in ['axis_angle', 'rotation_6d']
        assert from_rep != to_rep
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from_rep -> to_rep."""
        is_numpy = isinstance(x, np.ndarray)
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy() if x.is_cuda else x.numpy()

        # Flatten to (N, 3) or (N, 6)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        if self.from_rep == 'axis_angle' and self.to_rep == 'rotation_6d':
            # axis_angle (3,) -> rotation matrix (3,3) -> 6D (first two rows)
            rot = Rotation.from_rotvec(x_flat)
            mat = rot.as_matrix()  # (N, 3, 3)
            rot6d = mat[:, :2, :].reshape(mat.shape[0], 6)  # (N, 6)
        elif self.from_rep == 'rotation_6d' and self.to_rep == 'axis_angle':
            # 6D (first two rows) -> rotation matrix (3,3) -> axis_angle (3,)
            rot6d = x_flat  # (N, 6)
            mat = np.zeros((rot6d.shape[0], 3, 3), dtype=rot6d.dtype)
            mat[:, :2, :] = rot6d.reshape(rot6d.shape[0], 2, 3)
            # Compute third row via cross product to ensure orthogonality
            mat[:, 2, :] = np.cross(mat[:, 0, :], mat[:, 1, :])
            rot = Rotation.from_matrix(mat)
            axis_angle = rot.as_rotvec()  # (N, 3)
            rot6d = axis_angle
        else:
            raise ValueError(f"Unsupported conversion: {self.from_rep} -> {self.to_rep}")

        result = rot6d if self.from_rep == 'axis_angle' else rot6d
        result = result.reshape(orig_shape[:-1] + (result.shape[-1],))

        if is_numpy:
            return result
        else:
            return torch.from_numpy(result).float()

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert to_rep -> from_rep (inverse of forward)."""
        # Swap reps and call forward
        self.from_rep, self.to_rep = self.to_rep, self.from_rep
        result = self.forward(x)
        self.from_rep, self.to_rep = self.to_rep, self.from_rep
        return result

class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		# with torch.no_grad():
		# 	samples = self.base_policy(cond=cond, deterministic=True)
		# diffused_actions = (samples.trajectories.detach())
		# if return_numpy:
		# 	diffused_actions = diffused_actions.cpu().numpy()
		# return diffused_actions
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = getattr(samples, "trajectories", samples)
		if isinstance(diffused_actions, torch.Tensor):
			diffused_actions = diffused_actions.detach()
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions
	
class GenericDPWrapper:
    """
    Adapter for diffusion policies built from `extracted_config.yaml` style (e.g. diffusion_policy.*).

    Responsibilities:
    - ensure the policy receives `obs` shaped (B, n_obs_steps, obs_dim)
    - convert environment-format actions <-> policy-native format (axis-angle <-> rotation_6d)
    - call the policy's `predict_action` API and return (B, act_steps, action_dim) as numpy or tensor
    """

    def __init__(self, base_policy, cfg=None, device="cpu", env_action_dim=None):
        self.base_policy = base_policy
        self.cfg = cfg or {}
        self.device = device
        self.env_action_dim = env_action_dim  # expected action dim by the environment (e.g., 7 or 14)

        # buffer to accumulate last n_obs_steps observations for each parallel env
        self.obs_buffer = None
        self.n_obs_steps = int(getattr(self.cfg, "n_obs_steps", 2))

        # try to move model to device
        try:
            if hasattr(self.base_policy, "to"):
                self.base_policy = self.base_policy.to(self.device)
        except Exception:
            pass

        # set attributes the diffusion policies expect
        try:
            self.base_policy.device = self.device
        except Exception:
            pass
        try:
            self.base_policy.dtype = torch.float32
        except Exception:
            pass

        # rotation transformers (axis-angle <-> rotation_6d)
        self._rot_axis_to_6d = None
        self._rot_6d_to_axis = None
        try:
            # RotationTransformer implemented earlier in this file (scipy-based)
            self._rot_axis_to_6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
            self._rot_6d_to_axis = RotationTransformer(from_rep='rotation_6d', to_rep='axis_angle')
        except Exception:
            self._rot_axis_to_6d = None
            self._rot_6d_to_axis = None

    def _to_tensor(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype, device=self.device)
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def __call__(self, obs, initial_noise=None, return_numpy=True, deterministic=False):
        """
        obs: numpy array or tensor.
             Accepts either:
               - (B, obs_dim)  -> treated as current single step obs and stacked with buffer
               - (B, n_obs_steps, obs_dim) -> already stacked
        initial_noise: optional conditioning/action template (env-format)
        deterministic: best-effort deterministic sampling
        """
        # convert obs to tensor on device
        obs_t = self._to_tensor(obs)

        # Ensure obs_t is (B, To, Do)
        if obs_t.dim() == 2:
            # single-step current obs -> make it (B, 1, Do)
            obs_t = obs_t.unsqueeze(1)
        if obs_t.dim() != 3:
            raise RuntimeError(f"GenericDPWrapper: unexpected obs tensor shape {obs_t.shape}; expected (B,Do) or (B,To,Do)")

        B, To, Do = obs_t.shape

        # initialize obs buffer for this batch size if needed
        if self.obs_buffer is None or self.obs_buffer.shape[0] != B:
            # start with zeros and fill the last slot(s) with the first obs batch
            self.obs_buffer = torch.zeros(B, self.n_obs_steps, Do, device=self.device, dtype=torch.float32)
            # place the incoming obs at the end (so the first call uses last slot(s) = current obs)
            if To >= self.n_obs_steps:
                self.obs_buffer[:] = obs_t[:, -self.n_obs_steps:, :].to(self.device, dtype=torch.float32)
            else:
                self.obs_buffer[:, -To:, :] = obs_t.to(self.device, dtype=torch.float32)

        else:
            # shift and append current obs_t (handles To == 1 or larger)
            if To >= self.n_obs_steps:
                # latest To contains at least n_obs_steps; take last n_obs_steps directly
                self.obs_buffer = obs_t[:, -self.n_obs_steps:, :].to(self.device, dtype=torch.float32)
            else:
                # drop oldest `To` frames and append current `To` frames
                self.obs_buffer = torch.cat([self.obs_buffer[:, To:, :], obs_t.to(self.device, dtype=torch.float32)], dim=1)

        # use the stacked observations for the policy
        stacked_obs = self.obs_buffer  # (B, n_obs_steps, Do)

        # Deterministic: best-effort by seeding torch RNG locally
        if deterministic:
            torch_state = torch.get_rng_state()
            torch.manual_seed(0)

        # If an initial action (env-format) is provided, convert env->policy format if needed
        if initial_noise is not None:
            init_t = self._to_tensor(initial_noise, dtype=torch.float32)
            policy_action_dim = int(getattr(self.cfg, "action_dim", getattr(self.cfg, "act_dim", -1)))
            env_action_dim = self.env_action_dim
            if policy_action_dim and env_action_dim and init_t.dim() == 3 and init_t.shape[-1] == env_action_dim and policy_action_dim != env_action_dim:
                # single-arm 7 -> 10
                if env_action_dim == 7 and policy_action_dim == 10:
                    pos = init_t[..., :3]
                    rot_axis = init_t[..., 3:6]
                    grip = init_t[..., 6:7]
                    flat_rot = rot_axis.contiguous().view(-1, 3)
                    if self._rot_axis_to_6d is None:
                        raise RuntimeError("RotationTransformer (axis->6d) not available for env->policy conversion")
                    conv = self._rot_axis_to_6d.forward(flat_rot)
                    if isinstance(conv, np.ndarray):
                        conv = torch.from_numpy(conv)
                    conv = conv.to(device=self.device, dtype=torch.float32)
                    rot6 = conv.view(rot_axis.shape[0], rot_axis.shape[1], 6)
                    initial_noise = torch.cat([pos, rot6, grip], dim=-1)
                # dual-arm 14 -> 20
                elif env_action_dim == 14 and policy_action_dim == 20:
                    orig = init_t
                    B_, T_, _ = orig.shape
                    arm = orig.view(B_, T_, 2, 7)
                    out_arms = []
                    for a in range(2):
                        arm_a = arm[..., a, :]  # (B,T,7)
                        pos = arm_a[..., :3]
                        rot_axis = arm_a[..., 3:6]
                        grip = arm_a[..., 6:]
                        flat_rot = rot_axis.contiguous().view(-1, 3)
                        if self._rot_axis_to_6d is None:
                            raise RuntimeError("RotationTransformer (axis->6d) not available for env->policy conversion")
                        conv = self._rot_axis_to_6d.forward(flat_rot)
                        if isinstance(conv, np.ndarray):
                            conv = torch.from_numpy(conv)
                        conv = conv.to(device=self.device, dtype=torch.float32)
                        rot6 = conv.view(B_, T_, 6)
                        arm_conv = torch.cat([pos, rot6, grip], dim=-1)
                        out_arms.append(arm_conv)
                    initial_noise = torch.cat(out_arms, dim=-1)
                else:
                    # no rule for these dims; keep as-is
                    initial_noise = init_t
            else:
                initial_noise = init_t

        # call the canonical DP API using stacked_obs
        with torch.no_grad():
            if hasattr(self.base_policy, "predict_action"):
                obs_dict = {"obs": stacked_obs}
                result = self.base_policy.predict_action(obs_dict)
                if isinstance(result, dict) and "action" in result:
                    action_t = result["action"]
                else:
                    action_t = result
            else:
                # fallback: try calling the policy directly (DPPO-like)
                try:
                    result = self.base_policy(cond={"state": stacked_obs}, deterministic=True)
                    action_t = getattr(result, "trajectories", result)
                except Exception as e:
                    raise RuntimeError("GenericDPWrapper: unable to call base_policy. Inspect object API.") from e

        if deterministic:
            torch.set_rng_state(torch_state)

        # ensure torch tensor and correct device/dtype
        if isinstance(action_t, np.ndarray):
            action_t = torch.from_numpy(action_t).to(device=self.device, dtype=torch.float32)
        elif not isinstance(action_t, torch.Tensor):
            action_t = torch.tensor(action_t, device=self.device, dtype=torch.float32)
        else:
            action_t = action_t.to(device=self.device, dtype=torch.float32)

        # reshape to (B, act_steps, action_dim) if flattened
        if action_t.dim() == 2:
            act_steps = getattr(self.cfg, "n_action_steps", None) or getattr(self.cfg, "horizon", None)
            if act_steps is None and initial_noise is not None and isinstance(initial_noise, torch.Tensor):
                act_steps = initial_noise.shape[1]
            if act_steps is not None and action_t.shape[1] % act_steps == 0:
                action_dim = action_t.shape[1] // act_steps
                action_t = action_t.view(action_t.shape[0], act_steps, action_dim)

        # Convert policy-format rotation_6d -> env-format axis-angle if required
        if action_t.dim() == 3 and self.env_action_dim is not None:
            policy_act_dim = action_t.shape[-1]
            env_act_dim = self.env_action_dim
            # single arm 10 -> 7
            if policy_act_dim == 10 and env_act_dim == 7:
                pos = action_t[..., :3]
                rot6 = action_t[..., 3:9]
                grip = action_t[..., 9:10]
                flat_rot6 = rot6.contiguous().view(-1, 6)
                if self._rot_6d_to_axis is None:
                    raise RuntimeError("RotationTransformer (6d->axis) not available for policy->env conversion")
                conv = self._rot_6d_to_axis.forward(flat_rot6)
                if isinstance(conv, np.ndarray):
                    conv = torch.from_numpy(conv)
                conv = conv.to(device=self.device, dtype=torch.float32)
                axis = conv.view(rot6.shape[0], rot6.shape[1], 3)
                action_t = torch.cat([pos, axis, grip], dim=-1)
            # dual arm 20 -> 14
            elif policy_act_dim == 20 and env_act_dim == 14:
                B_, T_, _ = action_t.shape
                action_2 = action_t.view(B_, T_, 2, 10)
                out_arms = []
                for a in range(2):
                    arm = action_2[..., a, :]
                    pos = arm[..., :3]
                    rot6 = arm[..., 3:9]
                    grip = arm[..., 9:]
                    flat_rot6 = rot6.contiguous().view(-1, 6)
                    if self._rot_6d_to_axis is None:
                        raise RuntimeError("RotationTransformer (6d->axis) not available for policy->env conversion")
                    conv = self._rot_6d_to_axis.forward(flat_rot6)
                    if isinstance(conv, np.ndarray):
                        conv = torch.from_numpy(conv)
                    conv = conv.to(device=self.device, dtype=torch.float32)
                    axis = conv.view(rot6.shape[0], rot6.shape[1], 3)
                    out_arms.append(torch.cat([pos, axis, grip], dim=-1))
                action_t = torch.cat(out_arms, dim=-1)
            else:
                # no conversion needed or no rule; leave as-is
                pass

        if return_numpy:
            return action_t.detach().cpu().numpy()
        return action_t.detach()

    def reset_buffer(self):
        """Call this at environment reset to clear the observation buffer."""
        self.obs_buffer = None


# def load_base_policy(cfg):
# 	base_policy = hydra.utils.instantiate(cfg.model)
# 	base_policy = base_policy.eval()
# 	return DPPOBasePolicyWrapper(base_policy)

def load_base_policy(cfg):
	"""
	Try to instantiate either:
		- DPPO-style policy built with key `cfg.model`
		- diffusion-policy from `cfg.policy` (diffusion_policy.*)
	Returns a callable wrapper with signature (obs, initial_noise, return_numpy=True)
	"""
	device = getattr(cfg, "device", "cpu")

	# 2) diffusion_policy style (extracted_config.yaml)
	if getattr(cfg, "policy", None) is not None:
		dp_cfg = cfg.policy
		base_policy = hydra.utils.instantiate(dp_cfg)
		try:
			base_policy = base_policy.eval()
			if hasattr(base_policy, "to"):
				base_policy = base_policy.to(device)
		except Exception:
			try:
				base_policy = base_policy.eval()
			except Exception:
				pass

		# ensure expected attributes exist on policy
		try:
			base_policy.device = device
		except Exception:
			pass
		try:
			base_policy.dtype = torch.float32
		except Exception:
			pass
		
		import os
		# read checkpoint path from the policy config (fall back to network_path)
		ckpt_path = getattr(dp_cfg, "checkpoint_path", None) or getattr(dp_cfg, "network_path", None)
		if ckpt_path is None:
			print("No checkpoint path found in cfg.policy ('checkpoint_path' or 'network_path'); skipping weight load.")
			state_dict = None
		else:
			ckpt_path = os.path.expanduser(ckpt_path)
			print(f"Loading DP checkpoint from: {ckpt_path}")
			ckpt = torch.load(ckpt_path, map_location=device)

			# Normalize many checkpoint formats to a single `state_dict` mapping
			state_dict = None
			# (the rest of your existing state_dict extraction code continues here)
		if isinstance(ckpt, dict):
			# common container used in your checkpoint: ckpt['state_dicts'][...]
			if "state_dicts" in ckpt and isinstance(ckpt["state_dicts"], dict):
				sdicts = ckpt["state_dicts"]
				# prefer EMA then model
				for k in ("ema_model", "ema", "ema_state_dict", "model"):
					if k in sdicts:
						state_dict = sdicts[k]
						print(f"Using state_dict from state_dicts['{k}']")
						break
				# If keys in sdicts already look like flattened parameter keys
				if state_dict is None:
					# detect flattened layout: keys like "model.mid_modules.0..."
					first_key = next(iter(sdicts.keys()))
					if isinstance(first_key, str) and "." in first_key:
						state_dict = sdicts
						print("Using state_dicts (flat) as state_dict")
			# fallback to top-level keys
			if state_dict is None:
				for k in ("ema", "ema_model", "model", "state_dict", "state_dicts"):
					if k in ckpt:
						state_dict = ckpt[k]
						print(f"Using top-level checkpoint key '{k}' as state_dict")
						break
			# final fallback: use ckpt itself if it looks like a state dict
			if state_dict is None:
				# heuristically accept dicts that map param names to tensors (contain dots)
				first_key = next(iter(ckpt.keys()))
				if isinstance(first_key, str) and "." in first_key:
					state_dict = ckpt
					print("Treating entire checkpoint dict as state_dict")
		else:
			# checkpoint is not a dict - treat it as a raw state_dict
			state_dict = ckpt

		if state_dict is None:
			print("Warning: could not find a suitable state-dict inside checkpoint; skipping weight load.")
		else:
			# Helper that tries to load into a module; returns True on success
			def try_load(module, sd):
				try:
					module.load_state_dict(sd)
					print(f"Loaded weights into {module}")
					return True
				except Exception as e1:
					try:
						module.load_state_dict(sd, strict=False)
						print(f"Loaded weights into {module} with strict=False")
						return True
					except Exception as e2:
						return False

			loaded = False

			# 1) try loading directly into the instantiated policy
			try:
				if isinstance(state_dict, dict):
					loaded = try_load(base_policy, state_dict)
			except Exception:
				loaded = False

			# 2) try loading into base_policy.model if present
			if (not loaded) and hasattr(base_policy, "model"):
				try:
					loaded = try_load(base_policy.model, state_dict)
				except Exception:
					loaded = False

			# 3) if keys are prefixed like "model.xxx", try stripping the "model." prefix
			if (not loaded) and isinstance(state_dict, dict):
				def strip_prefix(sd, prefix="model."):
					return { (k[len(prefix):] if k.startswith(prefix) else k): v for k,v in sd.items() }
				stripped = strip_prefix(state_dict, prefix="model.")
				# try model first (common)
				if hasattr(base_policy, "model"):
					loaded = try_load(base_policy.model, stripped)
				# fallback to policy top-level
				if (not loaded):
					loaded = try_load(base_policy, stripped)

			# 4) final diagnostics if still not loaded
			if not loaded:
				print("Warning: Failed to load checkpoint weights into the policy.")
				# print a small diagnostics summary of keys to help debugging
				try:
					ck_keys = list(state_dict.keys())[:20]
					print("Sample checkpoint keys:", ck_keys)
				except Exception:
					pass
				try:
					if hasattr(base_policy, "model"):
						model_keys = list(base_policy.model.state_dict().keys())[:20]
						print("Sample base_policy.model state dict keys:", model_keys)
					else:
						policy_keys = list(base_policy.state_dict().keys())[:20]
						print("Sample base_policy state dict keys:", policy_keys)
				except Exception:
					pass

		# pass policy-level config so wrapper can read obs_dim, n_obs_steps, n_action_steps, etc.
		env_action_dim = getattr(cfg, "action_dim", None)
		return GenericDPWrapper(base_policy, cfg=dp_cfg, device=device, env_action_dim=env_action_dim)

	# 1) DPPO-style (existing behavior)
	if getattr(cfg, "model", None) is not None:
		base_policy = hydra.utils.instantiate(cfg.model)
		try:
			base_policy = base_policy.eval()
			if hasattr(base_policy, "to"):
				base_policy = base_policy.to(device)
		except Exception:
			try:
				base_policy = base_policy.eval()
			except Exception:
				pass
		return DPPOBasePolicyWrapper(base_policy)

	raise RuntimeError("load_base_policy: no 'model' or 'policy' found in cfg to instantiate base policy.")


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					agent.diffusion_policy.reset_buffer()
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	base_policy.reset_buffer()
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()