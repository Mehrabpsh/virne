"""Optional transactional PPO updates on a fixed, masked rollout distribution.

Clipping the surrogate is not a bound on policy divergence. This opt-in guard
checks the full rollout after a proposed Adam step and restores both model and
optimizer state before trying a smaller step. Rewards and action semantics are
unchanged; legacy configurations keep their existing update path.
"""
import copy
import json
import math
from pathlib import Path
import warnings

import numpy as np
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
    tolerance=settings.get('rollout_distribution_kl_atol',1e-6)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('rl.kl_guard.rollout_distribution_kl_atol must be finite and nonnegative')
    if config.training.get('distributed_training',False):
        raise ValueError('rl.kl_guard currently requires single-worker training')
    if config.nn.get('batch_norm',False) or config.nn.get('dropout_prob',0)!=0:
        raise ValueError('rl.kl_guard requires deterministic policy forwards: batch_norm=false and dropout_prob=0')


class PPOUpdateGuard:
    def __init__(self,solver,observations,actions,stored_logprobs,raw_observations=None):
        validate_settings(solver.config)
        self.solver=solver
        self.observations=observations
        self.target=float(solver.config.rl.target_kl)
        self.settings=solver.config.rl.kl_guard
        self.diagnostics=[]
        self.consistency={'batched_logprob_max_error':0.,'single_logprob_max_error':None,
                          'batching_max_state_kl':None,'single_recheck':False}
        with torch.no_grad():
            self.reference=self.distribution()
            recomputed=self.reference.log_prob(actions)
            error=(stored_logprobs-recomputed).abs().max()
            self.consistency['batched_logprob_max_error']=float(error)
            if not torch.isfinite(error) or error>self.settings.rollout_logprob_atol:
                # Collection forwards one graph at a time. A larger PyG batch
                # can change float32 reduction order, especially on CUDA. Verify
                # the collection path before diagnosing stale rollout data.
                replay=None
                if torch.isfinite(error) and raw_observations is not None:
                    self.consistency['single_recheck']=True
                    replay=torch.distributions.Categorical(logits=torch.cat([
                        self.distribution(solver.preprocess_obs(obs,actions.device)).logits
                        for obs in raw_observations],dim=0))
                    single_error=(stored_logprobs-replay.log_prob(actions)).abs().max()
                    self.consistency['single_logprob_max_error']=float(single_error)
                    _,batching_max=self.kl(replay,self.reference)
                    self.consistency['batching_max_state_kl']=batching_max
                    # Continue only when the original collection path agrees
                    # AND the complete action distributions differ negligibly.
                    if (torch.isfinite(single_error) and single_error<=self.settings.rollout_logprob_atol
                            and math.isfinite(batching_max)
                            and batching_max<=self.settings.get('rollout_distribution_kl_atol',1e-6)):
                        self.reference=replay
                        warnings.warn('PPO rollout verified by single-observation replay: '
                                      f'batch log-probability error={float(error):.6g}, '
                                      f'single error={float(single_error):.6g}, '
                                      f'max batching KL={batching_max:.6g}',RuntimeWarning)
                        return
                snapshot=self.save_mismatch(actions,stored_logprobs,raw_observations,replay)
                raise ValueError('Stored PPO log probabilities disagree before this rollout\'s optimizer steps: '
                                 f'max error {float(error):.6g}; consistency={self.consistency}; '
                                 f'check observation mutation, batching and policy mode; snapshot={snapshot}')

    def distribution(self,observations=None):
        observations=self.observations if observations is None else observations
        logits=self.solver.policy.act(observations)
        rl=self.solver.config.rl
        if rl.mask_actions and rl.maskable_policy and 'action_mask' in observations:
            logits=apply_mask_to_logit(logits,observations['action_mask'])
        return categorical_from_logits(logits,self.solver.softmax_temp)

    @staticmethod
    def kl(reference,latest):
        # Work in log space so tiny but valid action probabilities do not
        # underflow to zero before their log ratio is evaluated. Masked actions
        # have zero old mass and contribute zero, including -inf - -inf entries.
        delta=torch.where(reference.probs>0,reference.logits-latest.logits,0.)
        values=(reference.probs*delta).sum(-1).clamp_min(0)
        return float(values.mean()),float(values.max())

    def divergence(self):
        return self.kl(self.reference,self.distribution())

    def save_mismatch(self,actions,stored_logprobs,raw_observations,replay):
        """Keep the exact failing rollout rather than only an epoch checkpoint."""
        directory=getattr(self.solver,'log_dir',None)
        if directory is None:return None
        from omegaconf import OmegaConf
        def cpu(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
            if isinstance(value,np.ndarray):return torch.as_tensor(value.copy())
            if isinstance(value,np.generic):return value.item()
            if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
            if isinstance(value,(list,tuple)):return [cpu(item) for item in value]
            return value
        path=Path(directory)/'ppo_rollout_mismatch.pt'
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(cpu({'policy':self.solver.policy.state_dict(),
                'optimizer':self.solver.optimizer.state_dict(),
                'config':OmegaConf.to_container(self.solver.config,resolve=True),
                'observations':raw_observations,'actions':actions,'stored_logprobs':stored_logprobs,
                'batched_logits':self.reference.logits,
                'single_logits':None if replay is None else replay.logits,
                'consistency':self.consistency,'update_time':getattr(self.solver,'update_time',None),
                'training_epoch_id':getattr(self.solver,'training_epoch_id',None),
                'device':str(actions.device),'torch_version':str(torch.__version__),
                'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'float32_matmul_precision':torch.get_float32_matmul_precision(),
                'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_states':torch.cuda.get_rng_state_all() if actions.is_cuda else []}),path)
            path.with_suffix('.json').write_text(json.dumps(self.consistency,indent=2)+'\n')
            return str(path)
        except Exception as exc:
            warnings.warn(f'Could not save PPO mismatch snapshot: {exc}',RuntimeWarning)
            return None

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
