import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import os
from typing import Union
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

class RotationTransformer:
    """
    Torch-only minimal transformer that converts between:
      - axis-angle (rotation vector, shape (...,3)) and
      - rotation_6d (first two rows of rotation matrix flattened, shape (...,6))
    Keeps tensors on-device (no numpy/scipy roundtrips).
    """

    @staticmethod
    def axis_angle_to_rot6d(rotvec: torch.Tensor) -> torch.Tensor:
        # rotvec: (..., 3)

        # --- START OF PYTORCH3D LOGIC (Adapted from axis_angle_to_matrix) ---

        axis_angle = rotvec
        shape = axis_angle.shape
        device, dtype = axis_angle.device, axis_angle.dtype
        
        # 1. Angles / Magnitude
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)
        
        # 2. Skew-Symmetric Matrix (K)
        rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
        zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
        
        # PyTorch3D K (cross_product_matrix) construction:
        K = torch.stack(
            [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
        ).view(shape + (3,))
        
        K_sqrd = K @ K # K^2
        
        # 3. Coefficients and Final Matrix (R)
        identity = torch.eye(3, dtype=dtype, device=device)
        angles_sqrd = angles * angles
        angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd) # Handles theta=0
        
        # Rodrigues Formula: R = I + sinc(theta/pi)*K + ((1-cos(theta))/theta^2)*K^2
        R = (
            identity.expand(K.shape)
            + torch.sinc(angles / torch.pi) * K
            + ((1 - torch.cos(angles)) / angles_sqrd) * K_sqrd
        )

        # --- END OF PYTORCH3D LOGIC ---

        # 4. Final step: Convert Matrix (R) to 6D
        # The 6D representation is the first two rows flattened.
        # Note: R.shape is (..., 3, 3). We extract R[..., :2, :]
        batch_dim = R.size()[:-2]
        rot6 = R[..., :2, :].reshape(batch_dim + (6,))
        
        return rot6

    @staticmethod
    def rot6d_to_axis_angle(rot6: torch.Tensor) -> torch.Tensor:
        # rot6: (..., 6)

        # --- START OF PYTORCH3D LOGIC (Adapted from rotation_6d_to_matrix) ---
        d6 = rot6
        
        # 1. 6D -> Matrix (Gram-Schmidt process from PyTorch3D)
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1) # F.normalize from torch.nn.functional
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack((b1, b2, b3), dim=-2) # R is the 3x3 rotation matrix

        # --- END OF PYTORCH3D rotation_6d_to_matrix ---

        # --- START OF PYTORCH3D LOGIC (Adapted from matrix_to_axis_angle, fast=True) ---
        matrix = R
        
        # 2. Matrix -> Axis-Angle (PyTorch3D's direct formula)
        
        # Calculate omegas (axis vector components scaled by 2*sin(theta))
        omegas = torch.stack(
            [
                matrix[..., 2, 1] - matrix[..., 1, 2],
                matrix[..., 0, 2] - matrix[..., 2, 0],
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ],
            dim=-1,
        )
        norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
        traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
        
        # Calculate angle theta
        angles = torch.atan2(norms, traces - 1)

        zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
        omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles), atol=1e-8), zeros, omegas)

        near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)

        axis_angles = torch.empty_like(omegas)
        axis_angles[~near_pi] = (
            0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
        )

        # Handle theta near PI (requires matrix decomposition)
        # This section handles the singularity at pi/180 degrees where the trace is -1
        n = 0.5 * (
            matrix[near_pi][..., 0, :]
            + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
        )
        # Normalization factor for the axis at pi
        norm_n = torch.norm(n, dim=-1, keepdim=True)
        # Apply a small epsilon to prevent division by zero in case of zero vector
        norm_n = torch.where(norm_n == 0, 1e-8, norm_n) 
        
        axis_angles[near_pi] = angles[near_pi] * n / norm_n

        # --- END OF PYTORCH3D matrix_to_axis_angle (fast=True) ---

        return axis_angles

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

        # Rotation transformers (Torch implementation above)
        try:
            self._rot_axis_to_6d = RotationTransformer  # use static methods
            self._rot_6d_to_axis = RotationTransformer
        except Exception as e:
            raise RuntimeError("Failed to initialize RotationTransformer") from e

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
        if self.obs_buffer is None or self.obs_buffer.shape[0] != B:
            self.obs_buffer = torch.zeros(B, self.n_obs_steps, Do, device=self.device, dtype=torch.float32)
            if To >= self.n_obs_steps:
                # If we have enough incoming steps, use the last N steps (no change needed here)
                self.obs_buffer[:] = obs_t[:, -self.n_obs_steps:, :].to(self.device, dtype=torch.float32)
            else:
                # Case where To < self.n_obs_steps (e.g., first single step obs)
                
                # 1. Fill the tail with the available obs (this is the current step, To=1)
                current_obs_data = obs_t.to(self.device, dtype=torch.float32)
                self.obs_buffer[:, -To:, :] = current_obs_data 

                # 2. Duplicate the *oldest* incoming observation (the first one) to fill the buffer head.
                # The oldest incoming observation is at index 0 of the incoming obs_t.
                # In the single-step case (To=1), this is the current observation.
                padding_obs = current_obs_data[:, 0, :].unsqueeze(1) # (B, 1, Do)
                padding_length = self.n_obs_steps - To 
                
                # Expand the single observation across the padding length
                expanded_padding = padding_obs.expand(B, padding_length, Do)
                
                # Fill the head of the buffer with the expanded padding
                self.obs_buffer[:, :padding_length, :] = expanded_padding
        else:
            if To >= self.n_obs_steps:
                # replace whole buffer with last n_obs_steps of incoming obs
                self.obs_buffer[:] = obs_t[:, -self.n_obs_steps:, :].to(self.device, dtype=torch.float32)
            else:
                # shift left by To without allocating, then write
                self.obs_buffer[:, :self.n_obs_steps - To, :] = self.obs_buffer[:, To:, :]
                self.obs_buffer[:, self.n_obs_steps - To :, :] = obs_t.to(self.device, dtype=torch.float32)

        # use the stacked observations for the policy
        stacked_obs = self.obs_buffer  # (B, n_obs_steps, Do)

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
                        raise RuntimeError("RotationTransformer (axis->6d) not available")
                    # Call Torch transformer; ensure input on device
                    conv = self._rot_axis_to_6d.axis_angle_to_rot6d(flat_rot.to(self.device, dtype=torch.float32))
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
                        conv = self._rot_axis_to_6d.axis_angle_to_rot6d(flat_rot.to(self.device, dtype=torch.float32))
                        rot6 = conv.view(B_, T_, 6)
                        arm_conv = torch.cat([pos, rot6, grip], dim=-1)
                        out_arms.append(arm_conv)
                    initial_noise = torch.cat(out_arms, dim=-1)
                else:
                    # no rule for these dims; keep as-is
                    initial_noise = init_t
            else:
                initial_noise = init_t
            
            target_horizon = self.base_policy.horizon
            current_length = initial_noise.shape[1]
            Da = initial_noise.shape[-1]
            
            if current_length < target_horizon:
                padding_length = target_horizon - current_length
                padding = torch.zeros(
                	(B, padding_length, Da),
                	device=self.device,
                	dtype=initial_noise.dtype,
                )
                
                initial_noise = torch.cat([initial_noise, padding], dim=1)

        # call the canonical DP API using stacked_obs
        with torch.no_grad():
            if hasattr(self.base_policy, "predict_action"):
                obs_dict = {"obs": stacked_obs}
                # result = self.base_policy.predict_action(obs_dict=obs_dict)
                result = self.base_policy.predict_action(obs_dict=obs_dict, initial_trajectory=initial_noise)
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
                conv = self._rot_6d_to_axis.rot6d_to_axis_angle(flat_rot6.to(self.device, dtype=torch.float32))
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
                    conv = self._rot_6d_to_axis.rot6d_to_axis_angle(flat_rot6.to(self.device, dtype=torch.float32))
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
				success_count = 0
				attempt_count = 0
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					try:
						agent.diffusion_policy.reset_buffer()
					except:
						pass
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
					r_grid = np.stack(r)
					attempt_count += r_grid.shape[1]
					success_mask = np.any(r_grid > -8, axis=0)
					success_count += np.sum(success_mask)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				print(attempt_count, success_count, success_count/attempt_count)
				success_rate = np.mean(success)
				print(success_rate)
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
	try:
		base_policy.reset_buffer()
	except:
		pass
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