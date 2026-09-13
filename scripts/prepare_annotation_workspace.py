"""Prepare a preserved 25-case CT annotation workspace; never manufacture ground truth."""
from pathlib import Path
import argparse
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.configuration import configuration, load_configuration
from evaluation.annotation_workspace import prepare_workspace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset'))
    parser.add_argument('--existing-draft', type=Path, default=Path('eval_set'))
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not exist')
    parser.add_argument('--without-model-hints', action='store_true')
    parser.add_argument('--hint-config', type=Path, help='Optional additional saved detector configuration')
    args = parser.parse_args(argv)
    try:
        configs = [] if args.without_model_hints else [configuration(n) for n in ('baseline', 'contact', 'refined')]
        if args.hint_config and not args.without_model_hints:
            configs.append(load_configuration(args.hint_config))
        report = prepare_workspace(args.dataset, args.output_dir, existing_draft=args.existing_draft,
                                   hint_configs=configs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Prepared {len(report["cases"])} review cases. Reference labeling and anatomical review remain incomplete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
