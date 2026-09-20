"""Explicit, read-only-by-default entrypoint for the Σ-LeWM experiments."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from slewm.paths import REPO_ROOT
ROOT = REPO_ROOT

def registry():
    return json.loads((ROOT / 'slewm/registry/experiments.json').read_text())['experiments']

def audit():
    entries = registry()
    missing = []
    for e in entries:
        for p in e['entrypoints']:
            if not (ROOT / p).exists():
                missing.append({'experiment': e['id'], 'path': p})
    paths = []
    syntax = []
    for p in ROOT.rglob('*.py'):
        if '.git' in p.parts or '__pycache__' in p.parts:
            continue
        text = p.read_text()
        try:
            ast.parse(text)
        except SyntaxError as exc:
            syntax.append({'path': str(p.relative_to(ROOT)), 'error': str(exc)})
        for num, line in enumerate(text.splitlines(), 1):
            if re.search('/home/\\w+|/media/\\w+|/scratch-shared/|/gpfs/', line):
                paths.append({'file': str(p.relative_to(ROOT)), 'line': num})
    result = dict(experiment_families=len(entries), missing=missing, syntax_errors=syntax, machine_specific_paths=paths, note='File presence is not reproduction verification.')
    print(json.dumps({k: v for k, v in result.items() if k != 'machine_specific_paths'}, indent=2))
    print(f'Machine-specific path references: {len(paths)}')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    sub.add_parser('audit')
    ev = sub.add_parser('eval')
    ev.add_argument('--task', choices=['pusht', 'cube', 'tworoom', 'reacher'], required=True)
    ev.add_argument('--checkpoint', type=Path, required=True)
    ev.add_argument('--horizon', type=int, default=5)
    ev.add_argument('--cem', type=int, default=30)
    ev.add_argument('--seed', type=int, default=0)
    ev.add_argument('--episodes', type=int, default=50)
    ev.add_argument('--execute', action='store_true')
    toy = sub.add_parser('toy')
    toy.add_argument('--name', choices=['exact', 'pushing', 'alignment'], required=True)
    toy.add_argument('--execute', action='store_true')
    tr = sub.add_parser('train')
    tr.add_argument('--task', choices=['pusht', 'cube', 'tworoom', 'reacher', 'gear', 'franka'], required=True)
    tr.add_argument('--arm', choices=['lewm'] + [f'A{i}' for i in range(9)] + ['l0', 'c0', 'acmtm', 'smwm'], required=True)
    tr.add_argument('--epochs', type=int, required=True)
    tr.add_argument('--seed', type=int, default=3072)
    tr.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.command == 'list':
        for e in registry():
            print(f"{e['id']:28} {e['title']}")
        return
    if args.command == 'audit':
        result = audit()
        sys.exit(1 if result['missing'] or result['syntax_errors'] else 0)
    if args.command == 'train':
        if args.epochs < 1:
            p.error('epochs must be positive')
        if args.task in ('gear', 'franka'):
            if args.arm not in ('l0', 'c0', 'acmtm', 'smwm'):
                p.error('Isaac tasks use l0/c0/acmtm/smwm')
            cmd = [sys.executable, '-m', 'slewm.training.aligned', f'--config-name=paper_{args.task}_{args.arm}', f'seed={args.seed}', f'trainer.max_epochs={args.epochs}', 'trainer.min_epochs=0', 'early_stopping.enabled=false']
            if args.execute and (not os.environ.get('SLEWM_DATA_ROOT')):
                p.error('set SLEWM_DATA_ROOT for Isaac task data')
        else:
            data = dict(pusht='pusht', cube='ogb', tworoom='tworoom', reacher='dmc')[args.task]
            if args.arm == 'lewm':
                cmd = [sys.executable, '-m', 'slewm.training.aligned', f'data={data}', f'seed={args.seed}', f'trainer.max_epochs={args.epochs}', 'wandb.enabled=false']
            else:
                recipes = json.loads((ROOT / 'slewm/registry/training_recipes.json').read_text())
                if args.arm not in recipes:
                    p.error('Four-task arms use lewm or A0-A8')
                overrides = [s.replace('{DATA}', data).replace('{SEED}', str(args.seed)).replace('{EPOCHS}', str(args.epochs)) for s in recipes[args.arm]]
                cmd = [sys.executable, '-m', 'slewm.training.multistep'] + overrides
        if args.execute and (not os.environ.get('STABLEWM_HOME')):
            p.error('set STABLEWM_HOME before training')
    elif args.command == 'eval':
        if args.horizon < 1 or args.cem < 1 or args.episodes < 1:
            p.error('positive horizon/CEM/episodes required')
        cmd = [sys.executable, '-m', 'slewm.evaluation.planning', f'--config-name={args.task}', f'policy={args.checkpoint}', '+planner=cem', f'plan_config.horizon={args.horizon}', f'plan_config.receding_horizon={args.horizon}', f'eval.goal_offset_steps={5 * args.horizon}', f'eval.eval_budget={10 * args.horizon}', f'eval.num_eval={args.episodes}', f'solver.n_steps={args.cem}', 'solver.num_samples=300', f'seed={args.seed}']
        if args.execute and (not args.checkpoint.is_file() or not os.environ.get('STABLEWM_HOME')):
            p.error('execution requires an existing checkpoint and explicit STABLEWM_HOME')
    else:
        names = dict(exact='t1_exact_mi.py', pushing='t2_toy_world.py', alignment='t3_alignment.py')
        cmd = [sys.executable, 'scripts/toy/' + names[args.name]]
    print(json.dumps({'command': cmd, 'cwd': str(ROOT), 'execute': args.execute}, indent=2), flush=True)
    if args.execute:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join((str(ROOT), env.get('PYTHONPATH', '')))
        if args.command == 'train' and args.task in ('gear', 'franka'):
            env['LOCAL_DATASET_DIR'] = env['SLEWM_DATA_ROOT']
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
if __name__ == '__main__':
    main()
