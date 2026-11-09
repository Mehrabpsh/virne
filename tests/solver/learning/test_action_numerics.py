"""Regression tests for the archived PPO logit/masking failure."""
from types import SimpleNamespace
import pytest
import torch

from virne.solver.learning.rl_core.rl_solver import RLSolver
from virne.solver.learning.rl_core.searcher import select_action
from virne.solver.learning.utils import apply_mask_to_logit, categorical_from_logits


class FixedPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def act(self, observation):
        return self.logits.expand(observation['action_mask'].shape[0], -1)

    def evaluate(self, observation):
        return torch.zeros((observation['action_mask'].shape[0], 1))


def solver_with_logits(logits):
    solver = RLSolver.__new__(RLSolver)
    solver.policy = FixedPolicy(logits)
    solver.config = SimpleNamespace(rl=SimpleNamespace(mask_actions=True, maskable_policy=True))
    solver.softmax_temp = 1.0
    return solver


def test_archived_negative_logits_never_select_a_masked_action():
    # The real epoch-43 checkpoint emitted ~-135 million for valid actions.
    logits = torch.tensor([[-135020928., -135020928., -135020912.]])
    mask = torch.tensor([[True, False, True]])
    observation = {'action_mask': mask}
    solver = solver_with_logits(logits)
    for sample in [False, True]:
        action, _ = solver.select_action(observation, sample=sample)
        assert mask[0, action]
        action, _ = select_action(solver.policy, observation, mask=mask, sample=sample)
        assert mask[0, action]


def test_distribution_is_invariant_to_a_large_common_logit_offset():
    observation = {'action_mask': torch.tensor([[True, True, False]])}
    relative_logits = torch.tensor([[0., 1., 2.]])
    shifted_logits = relative_logits + 6_830_456.
    base = solver_with_logits(relative_logits)
    shifted = solver_with_logits(shifted_logits)
    action = torch.tensor([1])
    _, base_log_prob, base_entropy = base.evaluate_actions(observation, action)
    _, shifted_log_prob, shifted_entropy = shifted.evaluate_actions(observation, action)
    torch.testing.assert_close(shifted_log_prob, base_log_prob)
    torch.testing.assert_close(shifted_entropy, base_entropy)
    _, rollout_log_prob = shifted.select_action(observation, sample=False)
    torch.testing.assert_close(torch.as_tensor(rollout_log_prob), shifted_log_prob)


def test_rare_valid_action_keeps_its_log_probability_and_gradient():
    # Converting through probs clips log_prob at log(float32 epsilon),
    # although this valid action's true log probability is approximately -40.
    solver = solver_with_logits(torch.tensor([[0., -40., 0.]]))
    observation = {'action_mask': torch.tensor([[True, True, False]])}
    _, log_prob, _ = solver.evaluate_actions(observation, torch.tensor([1]))
    torch.testing.assert_close(log_prob, torch.tensor([-40.]))
    log_prob.sum().backward()
    assert solver.policy.logits.grad[0, 1] > 0.99


def test_masking_preserves_absolute_q_values():
    logits = torch.tensor([[-1.5e9, -1.4e9, 7.]])
    mask = torch.tensor([[True, True, False]])
    masked = apply_mask_to_logit(logits, mask)
    torch.testing.assert_close(masked[mask], logits[mask])
    assert masked.argmax(-1).item() == 1


@pytest.mark.parametrize('temperature', [0.7, 1., 1.5])
def test_rollout_and_update_likelihood_ratio_is_one_without_an_update(temperature):
    solver = solver_with_logits(torch.tensor([[6_830_456., 6_830_457., 6_830_458.]]))
    solver.softmax_temp = temperature
    observation = {'action_mask': torch.tensor([[True, True, False]])}
    action, rollout = solver.select_action(observation, sample=False)
    _, updated, entropy = solver.evaluate_actions(observation, torch.tensor([action]))
    torch.testing.assert_close(torch.exp(updated - torch.as_tensor(rollout)), torch.ones(1))
    assert torch.isfinite(entropy).all()


def test_masked_distribution_has_finite_gradients_and_zero_invalid_probability():
    logits = torch.tensor([[-1.5e9, -1.4e9, 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    distribution = categorical_from_logits(apply_mask_to_logit(logits, mask))
    assert distribution.probs[0, 2] == 0
    loss = -distribution.log_prob(torch.tensor([0])).mean() - .01 * distribution.entropy().mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


def test_empty_action_support_raises_instead_of_sampling_a_forbidden_action():
    with pytest.raises(ValueError, match='finite logit'):
        categorical_from_logits(torch.tensor([[-torch.inf, -torch.inf]]))


def test_common_logit_offset_does_not_receive_a_spurious_policy_gradient():
    offset = torch.tensor(0., requires_grad=True)
    logits = torch.tensor([[6_830_456., 6_830_457.]]) + offset
    distribution = categorical_from_logits(logits)
    (-distribution.log_prob(torch.tensor([1])).sum()).backward()
    torch.testing.assert_close(offset.grad, torch.tensor(0.), atol=1e-7, rtol=0)
