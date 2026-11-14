"""Check a failed DualGAT rollout at frozen weights, without training/simulation.

Snapshots with recorded behavior are checked against their original complete
distributions as well as fresh collection/recheck cycles. Older snapshots have
only selected-action probabilities, which cannot recover the full distribution.
"""
import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from virne.solver.learning.rl_policy.dual_gnn_policy import BiGatActorCritic
from virne.solver.learning.rl_core.policy_builder import OptimizerBuilder
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.rl_solver import RLSolver
from virne.solver.learning.rl_core.ppo_stability import PPOUpdateGuard,observation_digest
from virne.solver.learning.utils import apply_mask_to_logit,categorical_from_logits


def digest(policy):
    result=hashlib.sha256()
    for key,value in policy.state_dict().items():
        result.update(key.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run(snapshot,device,repeats,actor_dtype=None):
    torch.set_num_threads(1)
    s=torch.load(snapshot,map_location='cpu',weights_only=True)
    torch.set_float32_matmul_precision(s.get('float32_matmul_precision','highest'))
    torch.backends.cuda.matmul.allow_tf32=bool(s.get('cuda_matmul_allow_tf32',False))
    observations=s['observations'];first=observations[0];nn=s['config']['nn']
    solver=RLSolver.__new__(RLSolver)
    solver.config=OmegaConf.create(s['config']);solver.device=torch.device(device)
    dtype=actor_dtype or nn.get('actor_dtype','float32')
    solver.config.nn.actor_dtype=dtype
    solver.update_time=s['update_time'];solver.softmax_temp=1.
    solver.preprocess_obs=TensorConvertor.obs_as_tensor_for_dual_gnn
    solver.policy=BiGatActorCritic(p_net_num_nodes=first['p_net_x'].shape[0],
        p_net_x_dim=first['p_net_x'].shape[1],p_net_edge_dim=first['p_net_edge_attr'].shape[1],
        v_net_x_dim=first['v_net_x'].shape[1],v_net_edge_dim=first['v_net_edge_attr'].shape[1],
        embedding_dim=nn['embedding_dim'],num_layers=nn['num_gnn_layers'],
        dropout_prob=nn['dropout_prob'],batch_norm=nn['batch_norm'],actor_dtype=dtype).to(device)
    solver.policy.load_state_dict(s['policy'],strict=True);solver.policy.train()
    solver.optimizer=OptimizerBuilder.build_optimizer(solver.config,solver.policy)
    solver.optimizer.load_state_dict(s['optimizer'])
    before=digest(solver.policy);actions=s['actions'].to(device)
    batch=solver.preprocess_obs(observations,device)
    def distribution(obs):
        logits=solver.policy.act(obs)
        if solver.config.rl.mask_actions and solver.config.rl.maskable_policy:
            logits=apply_mask_to_logit(logits,obs['action_mask'])
        return categorical_from_logits(logits)
    checks=[]
    snapshot_check=None
    with torch.no_grad():
        original_error=float((distribution(batch).log_prob(actions)-s['stored_logprobs'].to(device)).abs().max())
        if s.get('behavior_records') is not None:
            try:
                guard=PPOUpdateGuard(solver,batch,actions,s['stored_logprobs'].to(device),
                    raw_observations=observations,behavior_records=s['behavior_records'])
                snapshot_check={'passed':True,**guard.consistency}
            except ValueError as exc:snapshot_check={'passed':False,'error':str(exc)}
        for _ in range(repeats):
            records=[]
            for obs in observations:
                dist=distribution(solver.preprocess_obs(obs,device))
                records.append({'logits':dist.logits.squeeze(0).cpu().clone(),
                                'update_time':solver.update_time,'observation_sha256':observation_digest(obs)})
            fresh=torch.stack([r['logits'] for r in records]).to(device).gather(-1,actions[:,None]).squeeze(-1)
            try:
                guard=PPOUpdateGuard(solver,batch,actions,fresh,raw_observations=observations,behavior_records=records)
                checks.append({'passed':True,**guard.consistency})
            except ValueError as exc:checks.append({'passed':False,'error':str(exc)})
    after=digest(solver.policy)
    return {'passed':all(c['passed'] for c in checks) and before==after
                    and (snapshot_check is None or snapshot_check['passed']),
            'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            'device':device,'torch_version':str(torch.__version__),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'actor_dtype':dtype,'policy_parameter_dtypes':solver.policy_parameter_dtypes(),
            'snapshot_epoch':s['training_epoch_id']+1,'snapshot_optimizer_updates':s['update_time'],
            'states':len(observations),'new_optimizer_updates':0,'policy_unchanged':before==after,
            'original_stored_vs_replay_logprob_error':original_error,
            'recorded_snapshot_check':snapshot_check,
            'fresh_collection_checks':checks,
            'limitation':'Frozen-weight inference only; dtype conversion is explicit and no optimizer steps run. Original full behavior is checked when recorded; older snapshots cannot recover missing distributions. This does not prove long-horizon convergence.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--actor-dtype',choices=['float32','float64'],
                        help='Override actor arithmetic; critic stays float32. Default: snapshot configuration.')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    if args.output.exists():parser.error('--output must be a new file')
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    result=run(args.snapshot,args.device,args.repeats,args.actor_dtype)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
