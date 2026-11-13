"""PPO safeguards: reject unsafe steps without corrupting Adam or policy state."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.ppo_stability import PPOUpdateGuard, validate_settings
from virne.solver.learning.rl_core.rl_solver import RLSolver
from virne.solver.learning.utils import categorical_from_logits, apply_mask_to_logit


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.tensor([[.1,.2,100.]]))

    def act(self,observations):return self.logits.expand(len(observations['action_mask']),-1)


def fixture(backtracks=8,lr=10.):
    config=OmegaConf.create({'rl':{'mask_actions':True,'maskable_policy':True,
        'target_kl':.02,'clip_grad':True,'max_grad_norm':1.,
        'kl_guard':{'enabled':True,'max_backtracks':backtracks,'backtrack_factor':.5,
                    'rollout_logprob_atol':.001}},'nn':{'batch_norm':False,'dropout_prob':0.},
        'training':{'distributed_training':False}})
    solver=RLSolver.__new__(RLSolver)
    solver.config=config;solver.policy=Policy();solver.softmax_temp=1.
    solver.optimizer=torch.optim.Adam(solver.policy.parameters(),lr=lr)
    observation={'action_mask':torch.tensor([[True,True,False],[True,True,False]])}
    actions=torch.tensor([0,1])
    with torch.no_grad():
        old=categorical_from_logits(apply_mask_to_logit(solver.policy.act(observation),observation['action_mask'])).log_prob(actions)
    return solver,observation,actions,old


def assert_state_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b


def clipped_actor_loss(solver,obs,actions,old):
    dist=categorical_from_logits(apply_mask_to_logit(solver.policy.act(obs),obs['action_mask']))
    ratio=torch.exp(dist.log_prob(actions)-old)
    advantage=torch.tensor([1.,-1.])
    return -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()


def test_clipping_alone_allows_large_policy_jump_but_guard_backtracks():
    plain,obs,actions,old=fixture()
    reference_guard=PPOUpdateGuard(plain,obs,actions,old)
    plain.update_grad(clipped_actor_loss(plain,obs,actions,old))
    assert reference_guard.divergence()[0]>1  # objective clipping is not a KL bound
    solver,obs,actions,old=fixture()
    reference=copy.deepcopy(solver.policy.state_dict())
    guard=PPOUpdateGuard(solver,obs,actions,old)
    # This is the ordinary actor gradient at ratio=1, where PPO clipping
    # applies no restriction to the first optimizer step.
    loss=clipped_actor_loss(solver,obs,actions,old)
    _,decision=guard.step(loss)
    assert decision['accepted'] and decision['backtracks']>0
    assert 0<decision['exact_kl']<=.02
    assert not torch.equal(solver.policy.logits,reference['logits'])
    assert solver.optimizer.param_groups[0]['lr']==10.
    # Retries must not accumulate multiple Adam timesteps/moment updates.
    assert solver.optimizer.state[solver.policy.logits]['step'].item()==1
    assert solver.policy.logits[0,2].item()==100.  # invalid action never gets policy gradient


def test_exhausted_guard_restores_initialized_adam_state_exactly():
    solver,obs,actions,old=fixture(backtracks=0,lr=.01)
    solver.update_grad(-solver.policy.logits[0,0])  # create real Adam moments
    solver.optimizer.param_groups[0]['lr']=10.
    with torch.no_grad():old=categorical_from_logits(apply_mask_to_logit(solver.policy.act(obs),obs['action_mask'])).log_prob(actions)
    model=copy.deepcopy(solver.policy.state_dict());optimizer=copy.deepcopy(solver.optimizer.state_dict())
    guard=PPOUpdateGuard(solver,obs,actions,old)
    _,decision=guard.step(-solver.policy.logits[0,0])
    assert not decision['accepted']
    assert_state_equal(solver.policy.state_dict(),model)
    assert_state_equal(solver.optimizer.state_dict(),optimizer)


def test_guard_reference_does_not_reset_between_minibatches():
    solver,obs,actions,old=fixture(lr=.1)
    guard=PPOUpdateGuard(solver,obs,actions,old)
    for _ in range(12):
        _,decision=guard.step(-solver.policy.logits[0,0])
        assert guard.divergence()[0]<=.02
        if not decision['accepted']:break
    assert len(guard.diagnostics)>1


def test_stale_rollout_logprob_fails_before_updating():
    solver,obs,actions,old=fixture()
    before=solver.policy.logits.detach().clone()
    with pytest.raises(ValueError,match='before this rollout'):
        PPOUpdateGuard(solver,obs,actions,old+1)
    torch.testing.assert_close(solver.policy.logits,before,rtol=0,atol=0)
    assert not solver.optimizer.state


def batching_fixture(bias=.002):
    solver,obs,actions,old=fixture()
    act=solver.policy.act
    def batched_act(observation):
        logits=act(observation)
        # Simulate a batch-size-dependent floating-point difference, without
        # changing policy weights or any collected observation.
        if len(observation['action_mask'])>1:
            logits=logits+torch.tensor([[bias,0.,0.]])
        return logits
    solver.policy.act=batched_act
    raw=[{'action_mask':mask.unsqueeze(0)} for mask in obs['action_mask']]
    solver.preprocess_obs=lambda observation,device:observation
    return solver,obs,actions,old,raw


def test_batch_roundoff_is_verified_with_collection_path_and_full_distribution():
    solver,obs,actions,old,raw=batching_fixture()
    with pytest.warns(RuntimeWarning,match='single-observation replay'):
        guard=PPOUpdateGuard(solver,obs,actions,old,raw_observations=raw)
    check=guard.consistency
    assert check['batched_logprob_max_error']>.001
    assert check['single_logprob_max_error']<1e-7
    assert check['batching_max_state_kl']<=1e-6
    torch.testing.assert_close(guard.reference.log_prob(actions),old)
    _,decision=guard.step(clipped_actor_loss(solver,obs,actions,old))
    assert decision['exact_kl']<=.02


def test_replay_does_not_excuse_stale_rollout_data_and_saves_exact_failure(tmp_path):
    solver,obs,actions,old,raw=batching_fixture()
    solver.log_dir=str(tmp_path)
    solver.update_time=17
    model=copy.deepcopy(solver.policy.state_dict())
    with pytest.raises(ValueError,match='before this rollout'):
        PPOUpdateGuard(solver,obs,actions,old+.1,raw_observations=raw)
    snapshot=torch.load(tmp_path/'ppo_rollout_mismatch.pt',weights_only=True)
    assert snapshot['update_time']==17
    assert snapshot['consistency']['single_logprob_max_error']>.001
    assert_state_equal(snapshot['policy'],model)
    assert_state_equal(solver.policy.state_dict(),model)
    assert not solver.optimizer.state


def test_material_batch_distribution_change_still_fails_even_when_replay_matches():
    solver,obs,actions,old,raw=batching_fixture(bias=.02)
    with pytest.raises(ValueError,match='before this rollout'):
        PPOUpdateGuard(solver,obs,actions,old,raw_observations=raw)
    assert not solver.optimizer.state


def test_replay_checks_unselected_actions_too():
    solver,obs,actions,old,raw=batching_fixture()
    # Collected action 0 has exactly the same probability, but probability
    # moves substantially between the two other valid actions. Comparing
    # selected-action log probabilities alone would miss that change.
    obs['action_mask'][:]=True
    for item in raw:item['action_mask'][:]=True
    solver.policy.act=lambda item:torch.log(torch.tensor(
        [[.2,.79,.01]] if len(item['action_mask'])==1 else [[.2,.01,.79]]
    )).expand(len(item['action_mask']),-1)
    actions=torch.tensor([0,0])
    # Force a tiny selected-action mismatch to trigger the collection replay.
    old=torch.full((2,),float(torch.log(torch.tensor(.2))))
    act=solver.policy.act
    solver.policy.act=lambda item:act(item)+(torch.tensor([[.002,0.,0.]])
        if len(item['action_mask'])>1 else 0.)
    with pytest.raises(ValueError,match='before this rollout'):
        PPOUpdateGuard(solver,obs,actions,old,raw_observations=raw)
    assert not solver.optimizer.state


@pytest.mark.parametrize('clip',[True,False])
def test_nonfinite_gradient_cannot_commit_an_optimizer_step(clip):
    solver,obs,actions,old=fixture()
    solver.config.rl.clip_grad=clip
    with torch.no_grad():solver.policy.logits[0,0]=0.
    before=solver.policy.logits.detach().clone()
    # Finite forward loss, infinite derivative at zero.
    with pytest.raises(RuntimeError,match='non-finite'):
        solver.update_grad(solver.policy.logits[0,0].sqrt())
    torch.testing.assert_close(solver.policy.logits,before,rtol=0,atol=0)
    assert not solver.optimizer.state


@pytest.mark.parametrize('path,value',[('rl.target_kl',None),('rl.target_kl',-1),
    ('rl.kl_guard.max_backtracks',-1),('rl.kl_guard.backtrack_factor',1.),
    ('rl.kl_guard.rollout_logprob_atol',-1.),('nn.batch_norm',True),
    ('rl.kl_guard.rollout_distribution_kl_atol',-1.),
    ('nn.dropout_prob',.5),('training.distributed_training',True)])
def test_invalid_guard_settings_fail_early(path,value):
    solver,_,_,_=fixture()
    OmegaConf.update(solver.config,path,value)
    with pytest.raises(ValueError):validate_settings(solver.config)


def test_training_mode_resumes_after_each_internal_validation():
    class Agent(InstanceAgent):
        def train(self):self.policy.train()
        def validate(self,env):
            self.policy.eval();self.validations+=1
        def learn_with_instance(self,instance):
            self.modes.append(self.policy.training)
            solution=type('Solution',(dict,),{'is_feasible':lambda self:True})({'v_net_r2c_ratio':1.})
            return solution,SimpleNamespace(logprobs=[np.array([0.])]),0.
        def merge_instance_experience(self,*args):pass
    agent=Agent(None);agent.policy=torch.nn.Linear(1,1);agent.validations=0;agent.modes=[]
    agent.config=OmegaConf.create({'training':{'if_use_random_training_seed':False,'save_interval':1,'eval_interval':1}})
    agent.seed=31;agent.logger=Mock();agent.logger.loggers={}
    agent.buffer=Mock();agent.buffer.size.return_value=0
    agent.target_steps=32;agent.rank=0;agent.save_model=Mock()
    env=Mock();env.v_net_simulator.num_v_nets=1
    env.step.return_value=({},0.,True,{'long_term_r2c_ratio':1.,'v_net_count':1})
    agent.learn_singly(env,num_epochs=3)
    assert agent.validations==2
    assert agent.modes==[True,True,True]
