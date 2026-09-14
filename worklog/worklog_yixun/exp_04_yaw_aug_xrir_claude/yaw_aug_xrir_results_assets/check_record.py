"""Recompute every recorded binding while preserving the original git HEAD."""
import argparse
import json
from pathlib import Path
from bind_provenance import collect, require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    report = json.loads(Path(parser.parse_args(argv).report).read_text())
    require(collect(head=report['git_HEAD'], **report['inputs']) == report, 'binding report mismatch')


if __name__ == '__main__':
    main()
