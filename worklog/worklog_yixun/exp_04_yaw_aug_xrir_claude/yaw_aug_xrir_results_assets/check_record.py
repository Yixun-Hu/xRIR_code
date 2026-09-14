"""Recompute every recorded binding while preserving the original git HEAD."""
import argparse
import json
from pathlib import Path
from tools.exp04_record import load_asset

binder = load_asset('bind_provenance')
collect, require = binder.collect, binder.require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    report = json.loads(Path(parser.parse_args(argv).report).read_text())
    require(collect(head=report['git_HEAD'], **report['inputs']) == report, 'binding report mismatch')


if __name__ == '__main__':
    main()
