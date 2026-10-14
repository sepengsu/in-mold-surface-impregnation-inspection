"""Re-audit verified saved OOF predictions into a separate directory, without fitting models."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from .protocol import load_frozen_splits, read_json, select_central_splits, sha256, smoke_subset
from .run import find_issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True, help='Existing experiment with completed OOF predictions.')
    parser.add_argument('--output', type=Path, required=True, help='New separate audit directory.')
    args = parser.parse_args(argv)
    source, destination = args.experiment.resolve(), args.output.resolve()
    config = read_json(source / 'experiment.json')
    settings = config['settings']
    data, split_file = Path(settings['data']).resolve(), Path(settings['splits']).resolve()
    if (destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination)
            or destination.is_relative_to(data) or data.is_relative_to(destination)):
        raise ValueError('Choose a new audit directory outside the source experiment and dataset.')
    print('Verifying saved OOF artifacts and current dataset...', flush=True)
    splits, digest = load_frozen_splits(data, split_file)
    if digest != config['manifest_sha256'] or sha256(split_file) != config['frozen_split_sha256']:
        raise ValueError('Dataset or frozen split differs from the OOF experiment.')
    splits = select_central_splits(splits,settings.get('central_only',False))
    if settings['smoke_test']:
        splits = smoke_subset(splits,central_only=settings.get('central_only',False))
    if splits != read_json(source / 'splits.json')['splits']:
        raise ValueError('Saved OOF experiment split differs from the validated dataset.')
    audit_args = SimpleNamespace(data=data, smoke_test=settings['smoke_test'])
    find_issues(audit_args, source, splits, destination=destination)
    print(f'Audit complete: {destination / "index.html"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
