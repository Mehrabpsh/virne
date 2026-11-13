"""Optional transactional PPO updates on a fixed, masked rollout distribution.

Clipping the surrogate is not a bound on policy divergence. This opt-in guard
checks the full rollout after a proposed Adam step and restores both model and
optimizer state before trying a smaller step. Rewards and action semantics are
unchanged; legacy configurations keep their existing update path.
"""
import copy
import math

import torch

from ..utils import apply_mask_to_logit, categorical_from_logits


def validate_settings(config):
    settings=config.rl.get('kl_guard',{})
    if not isinstance(settings.get('enabled',False),bool):
        raise ValueError('rl.kl_guard.enabled must be boolean')
    if not settings.get('enabled',False):return
    target=config.rl.get('target_kl')
    if isinstance(target,bool) or not isinstance(target,(int,float)) or not math.isfinite(target) or target<=0:
        raise ValueError('rl.kl_guard requires a finite positive rl.target_kl')
    count=settings.get('max_backtracks',4)
    if isinstance(count,bool) or not isinstance(count,int) or count<0:
        raise ValueError('rl.kl_guard.max_backtracks must be a nonnegative integer')
    factor=settings.get('backtrack_factor',.5)
    if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not 0<factor<1:
        raise ValueError('rl.kl_guard.backtrack_factor must lie strictly between zero and one')
    tolerance=settings.get('rollout_logprob_atol',1e-3)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('rl.kl_guard.rollout_logprob_atol must be finite and nonnegative')
    if config.training.get('distributed_training',False):
        raise ValueError('rl.kl_guard currently requires single-worker training')
    if config.nn.get('batch_norm',False) or config.nn.get('dropout_prob',0)!=0:
        raise ValueError('rl.kl_guard requires deterministic policy forwards: batch_norm=false and dropout_prob=0')


class PPOUpdateGuard:
    def __init__(self,solver,observations,actions,stored_logprobs):
        validate_settings(solver.config)
        self.solver=solver
        self.observations=observations
        self.target=float(solver.config.rl.target_kl)
        self.settings=solver.config.rl.kl_guard
        self.diagnostics=[]
        with torch.no_grad():
            self.reference=self.distribution()
            recomputed=self.reference.log_prob(actions)
            error=(stored_logprobs-recomputed).abs().max()
            if not torch.isfinite(error) or error>self.settings.rollout_logprob_atol:
                raise ValueError('Stored PPO log probabilities disagree before any optimizer step: '
                                 f'max error {float(error):.6g}; check observation mutation, batching and policy mode')

    def distribution(self):
        logits=self.solver.policy.act(self.observations)
        rl=self.solver.config.rl
        if rl.mask_actions and rl.maskable_policy and 'action_mask' in self.observations:
            logits=apply_mask_to_logit(logits,self.observations['action_mask'])
        return categorical_from_logits(logits,self.solver.softmax_temp)

    def divergence(self):
        latest=self.distribution()
        # Work in log space so tiny but valid action probabilities do not
        # underflow to zero before their log ratio is evaluated. Masked actions
        # have zero old mass and contribute zero, including -inf - -inf entries.
        delta=torch.where(self.reference.probs>0,self.reference.logits-latest.logits,0.)
        values=(self.reference.probs*delta).sum(-1).clamp_min(0)
        return float(values.mean()),float(values.max())

    def step(self,loss):
        solver=self.solver
        model=copy.deepcopy(solver.policy.state_dict())
        optimizer=copy.deepcopy(solver.optimizer.state_dict())
        rates=[group['lr'] for group in solver.optimizer.param_groups]
        scale=1.
        grad_norm=None
        try:
            for attempt in range(self.settings.max_backtracks+1):
                if attempt==0:
                    grad_norm=solver.update_grad(loss)
                else:
                    # The clipped gradients still correspond to the restored
                    # pre-step parameters. Adam moments/counters must be restored
                    # too; loading a fresh copy avoids aliasing the snapshot.
                    solver.policy.load_state_dict(model)
                    solver.optimizer.load_state_dict(copy.deepcopy(optimizer))
                    scale=self.settings.backtrack_factor**attempt
                    for group,rate in zip(solver.optimizer.param_groups,rates):group['lr']=rate*scale
                    solver.optimizer.step()
                with torch.no_grad():
                    try:mean,maximum=self.divergence()
                    except ValueError:mean=maximum=float('inf')
                if math.isfinite(mean) and math.isfinite(maximum) and mean<=self.target:
                    result={'accepted':True,'exact_kl':mean,'max_state_kl':maximum,
                            'backtracks':attempt,'step_scale':scale}
                    self.diagnostics.append(result)
                    return grad_norm,result
            solver.policy.load_state_dict(model)
            solver.optimizer.load_state_dict(copy.deepcopy(optimizer))
            with torch.no_grad():mean,maximum=self.divergence()
            result={'accepted':False,'exact_kl':mean,'max_state_kl':maximum,
                    'backtracks':self.settings.max_backtracks,'step_scale':0.}
            self.diagnostics.append(result)
            return grad_norm,result
        except BaseException:
            solver.policy.load_state_dict(model)
            solver.optimizer.load_state_dict(copy.deepcopy(optimizer))
            raise
        finally:
            for group,rate in zip(solver.optimizer.param_groups,rates):group['lr']=rate
