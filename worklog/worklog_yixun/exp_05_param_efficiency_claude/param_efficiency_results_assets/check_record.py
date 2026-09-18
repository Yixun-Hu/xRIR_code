"""Recompute every recorded exp_05 binding while preserving the original git HEAD."""
import argparse
import json
import re
from pathlib import Path

from tools.exp05_record import load_asset

binder = load_asset('bind_provenance')
collect, require = binder.collect, binder.require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',
                        help='Directory holding the timestamped binding reports; verify the latest')
    directory = Path(parser.parse_args(argv).directory)
    reports = sorted(path for path in directory.glob('binding_report_*.json')
                     if re.fullmatch(r'binding_report_\d{8}T\d{12}Z.json', path.name))
    require(reports, 'no timestamped binding reports')
    report = json.loads(reports[-1].read_text())
    require(collect(head=report['git_HEAD'], **report['inputs']) == report,
            'binding report mismatch')


if __name__ == '__main__':
    main()
