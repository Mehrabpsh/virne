"""Preserve action-score differences before categorical normalization."""
import copy

import pytest
import torch
from torch_geometric.data import Batch, Data

from virne.solver.learning.rl_policy.dual_gnn_policy import BiGatActorCritic
from virne.solver.learning.utils import categorical_from_logits, apply_mask_to_logit


def inputs():
    def graph(nodes):
        source=torch.arange(nodes-1)
        edge_index=torch.stack([torch.cat([source,source+1]),torch.cat([source+1,source])])
        return Data(x=torch.randn(nodes,4),edge_index=edge_index,
                    edge_attr=torch.ones(edge_index.shape[1],1))
    return {'p_net':Batch.from_data_list([graph(6),graph(6)]),
            'v_net':Batch.from_data_list([graph(3),graph(4)]),
            'curr_v_node_id':torch.tensor([0,1]),
            'action_mask':torch.tensor([[True,True,False,True,True,True]]*2)}


def policy(dtype):
    return BiGatActorCritic(p_net_num_nodes=6,p_net_x_dim=4,p_net_edge_dim=1,
        v_net_x_dim=4,v_net_edge_dim=1,embedding_dim=8,num_layers=2,
        dropout_prob=0.,batch_norm=False,actor_dtype=dtype)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_actor_precision_keeps_critic_and_shared_inputs_unchanged(dtype):
    torch.manual_seed(31)
    model=policy(dtype);obs=inputs()
    saved={key:obs[key].clone() for key in ('p_net','v_net')}
    logits=model.act(obs)
    assert logits.dtype==getattr(torch,dtype)
    values=model.evaluate(obs)
    assert values.dtype==torch.float32
    for key in saved:
        assert obs[key].x.dtype==obs[key].edge_attr.dtype==torch.float32
        assert obs[key].edge_index.dtype==obs[key].batch.dtype==torch.int64
        for attribute in ('x','edge_attr','edge_index','batch'):
            torch.testing.assert_close(obs[key][attribute],saved[key][attribute],rtol=0,atol=0)
    dist=categorical_from_logits(apply_mask_to_logit(logits,obs['action_mask']))
    assert (dist.probs[:,2]==0).all()
    loss=-dist.log_prob(torch.tensor([0,1])).mean()+values.square().mean()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.zero_grad();loss.backward();optimizer.step()
    for module,expected in [(model.actor,getattr(torch,dtype)),(model.critic,torch.float32)]:
        gradients=[parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(gradient.dtype==expected and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum()>0 for gradient in gradients)


def test_float64_actor_preserves_relative_scores_inside_large_linear_projection():
    torch.manual_seed(37)
    precise=policy('float64');obs=inputs()
    # Same mathematical network and same weights; float32 loses differences
    # when the final linear projection rounds a large common output offset.
    ordinary=copy.deepcopy(precise).float()
    with torch.no_grad():
        baseline=categorical_from_logits(precise.act(obs)).logits
        precise.actor.encoder.lin[-1].bias.add_(1e9)
        ordinary.actor.encoder.lin[-1].bias.add_(1e9)
        high=categorical_from_logits(precise.act(obs)).logits
        low=categorical_from_logits(ordinary.act(obs)).logits
    torch.testing.assert_close(high,baseline,atol=2e-7,rtol=0)
    assert float((low-baseline).abs().max())>1e-4


def test_unsupported_actor_precision_is_rejected():
    with pytest.raises(ValueError,match='actor_dtype must be float32 or float64'):
        policy('float16')
