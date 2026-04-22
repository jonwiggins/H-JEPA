"""
Cross-Entropy Method (CEM) planner for latent-space MPC.

Plans action sequences in the latent space of a JEPA-style world model. Given
an encoder, an action-conditioned predictor, an initial observation, and a
goal observation, the planner searches for an action sequence that drives
the predicted latent trajectory toward the goal latent.

Reference: LeWorldModel (https://arxiv.org/abs/2603.19312)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class CEMPlanner:
    """
    CEM-based MPC planner that operates entirely in latent space.

    The planner samples ``num_samples`` action sequences of length ``horizon``
    from a Gaussian over actions, rolls each sequence through the latent
    dynamics, scores them by goal-matching cost, retains the top ``num_elites``,
    refits the Gaussian to those elites, and iterates ``num_iterations`` times.
    The first action of the best sequence is returned (MPC-style — replan
    after each environment step).

    Args:
        encoder: Module mapping pixel observations to latent embeddings.
            Must accept tensor of shape ``[B, *obs_shape]`` and return ``[B, D]``.
        predictor: Module mapping ``(embeddings [B, T, D], actions [B, T, A])``
            to next-step embeddings ``[B, T, D]``.
        action_dim: Dimensionality of the action vector.
        horizon: Planning horizon (number of actions to optimize).
        num_samples: Sampled action sequences per CEM iteration.
        num_elites: Number of top-scoring samples used to refit the Gaussian.
        num_iterations: CEM iterations per planning call.
        action_low: Lower bound for actions (broadcastable to [action_dim]).
        action_high: Upper bound for actions (broadcastable to [action_dim]).
        cost_fn: Optional cost function ``(pred_traj, goal_emb) -> [num_samples]``
            scoring each rolled-out trajectory. Defaults to terminal MSE
            against the goal embedding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 300,
        num_elites: int = 30,
        num_iterations: int = 30,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        if num_elites > num_samples:
            raise ValueError(f"num_elites ({num_elites}) cannot exceed num_samples ({num_samples})")

        self.encoder = encoder
        self.predictor = predictor
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.cost_fn = cost_fn if cost_fn is not None else self._terminal_goal_cost

    @staticmethod
    def _terminal_goal_cost(
        predicted_trajectory: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Terminal MSE goal-matching cost: ||ẑ_H - z_g||² for each sampled rollout.

        Args:
            predicted_trajectory: [num_samples, horizon, D]
            goal_embedding: [D]

        Returns:
            [num_samples] cost values (lower is better).
        """
        terminal = predicted_trajectory[:, -1, :]  # [num_samples, D]
        return ((terminal - goal_embedding.unsqueeze(0)) ** 2).sum(dim=-1)

    @torch.no_grad()
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a single observation to a [D]-shaped latent."""
        if obs.ndim == 3:  # [C, H, W]
            obs = obs.unsqueeze(0)
        emb = self.encoder(obs)  # [1, D] or [1, N, D]
        if emb.ndim == 3:
            # If encoder returns tokens, pool to a single vector
            emb = emb.mean(dim=1)
        return emb.squeeze(0)  # type: ignore[no-any-return]

    @torch.no_grad()
    def _rollout(
        self,
        initial_embedding: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Roll a batch of action sequences through the latent dynamics.

        Args:
            initial_embedding: [D] starting latent.
            action_sequences: [num_samples, horizon, action_dim]

        Returns:
            [num_samples, horizon, D] predicted latent trajectory. Index t is
            the predicted state after applying action t.
        """
        B = action_sequences.shape[0]
        T = action_sequences.shape[1]
        D = initial_embedding.shape[0]

        # Build embedding sequence by rolling forward one step at a time.
        # The predictor expects an embedding sequence + action sequence and
        # returns next-step predictions in parallel via causal attention.
        # For closed-loop rollout we feed the initial embedding plus the
        # autoregressively predicted embeddings.
        embeddings = torch.zeros(
            B, T, D, device=initial_embedding.device, dtype=initial_embedding.dtype
        )
        embeddings[:, 0, :] = initial_embedding.unsqueeze(0).expand(B, -1)

        for t in range(T - 1):
            # Predict ẑ_{t+1} from (z_0, ..., z_t) and (a_0, ..., a_t).
            # Use prefix of length t+1.
            prefix_embs = embeddings[:, : t + 1, :]
            prefix_acts = action_sequences[:, : t + 1, :]
            preds = self.predictor(prefix_embs, prefix_acts)
            embeddings[:, t + 1, :] = preds[:, -1, :]

        # The "predicted trajectory" is positions 1..T (states reached after each action).
        # For terminal-cost purposes we want the state after the final action,
        # which we obtain with one more predictor call.
        full_preds = self.predictor(embeddings, action_sequences)
        # Return the full predicted next-states sequence.
        return full_preds  # type: ignore[no-any-return]

    @torch.no_grad()
    def plan(
        self,
        initial_observation: torch.Tensor,
        goal_observation: torch.Tensor,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Run CEM to find a good action sequence and return the first action.

        Args:
            initial_observation: Pixel observation of the starting state.
            goal_observation: Pixel observation of the goal state.
            return_full_sequence: If True, return the full optimized action
                sequence ``[horizon, action_dim]`` instead of just the first
                action ``[action_dim]``.

        Returns:
            Best first action ``[action_dim]`` (default) or the full sequence.
        """
        device = next(self.encoder.parameters()).device

        # Planning runs the encoder/predictor in eval mode so that BatchNorm
        # uses running statistics (allowing single-sample encoding) and
        # dropout is disabled. The original training mode is restored after.
        encoder_was_training = self.encoder.training
        predictor_was_training = self.predictor.training
        self.encoder.eval()
        self.predictor.eval()
        try:
            z_init = self._encode(initial_observation.to(device))
            z_goal = self._encode(goal_observation.to(device))
            result = self._run_cem(z_init, z_goal, device, return_full_sequence)
        finally:
            if encoder_was_training:
                self.encoder.train()
            if predictor_was_training:
                self.predictor.train()
        return result

    @torch.no_grad()
    def _run_cem(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        device: torch.device,
        return_full_sequence: bool,
    ) -> torch.Tensor:

        # Initialize the action distribution as a uniform prior centered in the action range.
        action_low = torch.as_tensor(self.action_low, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(self.action_high, device=device, dtype=torch.float32)
        if action_low.ndim == 0:
            action_low = action_low.expand(self.action_dim)
        if action_high.ndim == 0:
            action_high = action_high.expand(self.action_dim)

        mean = ((action_low + action_high) / 2).unsqueeze(0).expand(self.horizon, -1).clone()
        std = ((action_high - action_low) / 2).unsqueeze(0).expand(self.horizon, -1).clone()

        for _ in range(self.num_iterations):
            # Sample action sequences from current Gaussian.
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise  # [N, H, A]
            samples = samples.clamp(action_low, action_high)

            # Roll out and score.
            predicted_trajectory = self._rollout(z_init, samples)
            costs = self.cost_fn(predicted_trajectory, z_goal)  # [N]

            # Refit to elites.
            elite_idx = torch.topk(costs, self.num_elites, largest=False).indices
            elites = samples[elite_idx]  # [num_elites, H, A]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=1e-3)

        if return_full_sequence:
            return mean
        return mean[0]
