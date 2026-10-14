"""Inspect the actual Virne lifecycle, rather than trusting registry labels."""
from ..config.schema import ConfigError, require


def inspect_solver(name, requested='auto'):
    from virne.solver import Solver, SolverRegistry
    from virne.solver.learning.rl_core import RLSolver
    try:
        cls=SolverRegistry.get(name)
    except NotImplementedError as exc:
        raise ConfigError(f'Unknown Virne solver {name!r}') from exc
    solve=callable(getattr(cls,'solve',None)) and cls.solve is not Solver.solve
    rl=issubclass(cls,RLSolver)
    # RLSolver.learn delegates to learn_singly; merely inheriting learn is insufficient.
    training=rl and callable(getattr(cls,'learn',None)) and callable(getattr(cls,'learn_singly',None))
    checkpoint=rl and all(callable(getattr(cls,k,None)) for k in ('save_model','load_model'))
    kind=('learned' if rl else 'heuristic') if requested=='auto' else requested
    require(solve,f'{name}: no implemented solve(instance) API')
    if kind=='learned':
        require(training and checkpoint,f'{name}: unsupported learned execution contract; requires RLSolver learn_singly/save_model/load_model lifecycle')
    else:
        require(not rl,f'{name}: RLSolver cannot be requested as a heuristic; would evaluate an untrained model')
    return {'kind':kind,'class':cls.__module__+'.'+cls.__name__,'registry_type':cls.type,
            'evaluation_only':not rl,'train_and_evaluate':training,'checkpoint_save_load':checkpoint,
            'online_evaluation_supported':solve,'evidence':'Static lifecycle/API compatibility; constructor requirements validated where available, execution still verified at runtime',
            'warning':'Registry learning label does not imply trainable/checkpointable; evaluation-only search solver' if cls.type=='r_learning' and not rl else None}
