"""Run the deferred GPU parity cases and publish a receipt of exactly what they did.

Plan section 3: the bit-identity gate (four trainer cases) and the evaluator parity
(five cases) are GPU-only and are deferred until a device is free; they MUST pass before
any launch.  A log is not evidence of that -- a log saying ``9 failed`` is still a log --
so this entry point owns the run AND its evidence: it reserves the log and the JUnit XML
exclusively before spawning pytest (an existing one at either path is refused, never
reused), clears the ambient pytest options, executes exactly the nine registered node ids
under that one pytest, refuses anything that is missing, skipped, failed, unregistered or
that left the reserved XML unwritten, and writes an immutable receipt naming each case
with the outcome PARSED FROM THAT XML, the
pytest exit status, the reviewed commit the checkout is at, the CUDA device the cases ran
on and the digests of the log and the JUnit XML.  The record binder takes that receipt as
``--evidence gpu_parity=<receipt>`` and re-checks all of it, including that the checkout
was clean (``--allow-dirty`` exists for rehearsals and is refused as record evidence).

    python tools/exp07_parity.py --log <record>/gpu_parity_<commit>.log \\
        --reviewed-commit <commit> --gpu 1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from tools import provenance as p

REPO = Path(__file__).resolve().parents[1]
TRAINER = 'tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer'
EVAL_GPU = 'tests/test_exp07_eval_gpu.py::'
# The registered parity set (Codex round-6 review, "Deferred GPU tests"): every id is
# checked against pytest's own collection by tests/test_exp07_parity.py.
TESTS = tuple([TRAINER + '[{}-{}-32]'.format(yaw, backbone)
               for yaw in ('False', 'True') for backbone in ('simple', 'cylindrical')] +
              [EVAL_GPU + 'test_unseen_32_query_run_is_bit_identical_to_exp04'] +
              [EVAL_GPU + name + '[{}]'.format(shot)
               for name in ('test_seen_final_batch_matches_a_direct_frozen_call',
                            'test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions')
               for shot in (8, 1)])
OUTCOMES = {'failure': 'failed', 'error': 'error', 'skipped': 'skipped'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cuda_device():
    """The device the parity cases will run on; without one they would only skip."""
    import torch
    require(torch.cuda.is_available(), 'no CUDA device is visible: parity cannot run')
    return torch.cuda.get_device_name(torch.cuda.current_device())


def command(junit):
    # -o addopts= together with the cleared PYTEST_ADDOPTS in run(): no ambient option may
    # turn this into a collection, a help screen or a different set of cases.
    return [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'addopts=',
            '--junitxml=' + str(junit)] + list(TESTS)


def outcomes(junit):
    """Node id -> outcome, read from pytest's own JUnit XML rather than from the log."""
    result = {}
    for case in ElementTree.parse(str(junit)).getroot().iter('testcase'):
        name = case.get('file') or (case.get('classname', '').replace('.', '/') + '.py')
        node = name + '::' + case.get('name')
        outcome = 'passed'
        for child in case:
            outcome = OUTCOMES.get(child.tag, outcome)
        require(node not in result, 'duplicate test case: ' + node)
        result[node] = outcome
    return result


def parsed_outcomes(junit):
    """:func:`outcomes`, refusing a file that is not pytest's own XML at all."""
    try:
        return outcomes(junit)
    except ElementTree.ParseError as error:
        raise ValueError('the parity JUnit XML is not parseable: ' + str(error))


def reserve(*paths):
    """Claim every output path exclusively, or create none of them; return the earliest mtime.

    ``O_CREAT | O_EXCL`` before pytest is spawned is what makes the evidence this
    invocation's own: a file left by an earlier run -- or planted by someone who wants a
    passing receipt without a run -- is refused instead of reused or overwritten.
    """
    created, times = [], []
    try:
        for path in paths:
            os.close(os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            created.append(Path(path))
            times.append(created[-1].stat().st_mtime)
    except BaseException:
        for item in created:
            item.unlink()
        raise
    return min(times)


def check(recorded, status):
    """Exactly the nine registered cases, all passed, under a successful pytest exit."""
    missing = [node for node in TESTS if node not in recorded]
    require(not missing, 'missing parity cases: ' + ', '.join(missing))
    unregistered = sorted(set(recorded) - set(TESTS))
    require(not unregistered, 'unregistered parity cases: ' + ', '.join(unregistered))
    refused = sorted(node for node in TESTS if recorded[node] != 'passed')
    require(not refused, 'parity cases did not pass: ' + ', '.join(
        '{} ({})'.format(node, recorded[node]) for node in refused))
    require(status == 0, 'pytest exited ' + str(status))


def run(log, junit, gpu, runner):
    """Run the nine cases once, into output paths this invocation reserved for itself.

    Both the log and the JUnit XML are created exclusively BEFORE pytest is spawned, and
    the XML the child leaves behind must be newer than that reservation and non-empty, so
    a run that collected nothing -- the child was turned into ``--help``, say -- cannot be
    receipted against someone else's passing XML.
    """
    reserved = reserve(log, junit)
    environment = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1',
                       PYTHONHASHSEED='0', OMP_NUM_THREADS='2', PYTEST_ADDOPTS='')
    if gpu is not None:
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    argv = command(junit)
    completed = runner(argv, cwd=str(REPO), env=environment, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    output = completed.stdout if isinstance(completed.stdout, bytes) else b''
    with open(str(log), 'r+b') as stream:  # the reserved, still empty, log of this run
        stream.write(output)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    print(output.decode('utf-8', 'replace'), flush=True)
    written = Path(junit).stat()
    require(written.st_size > 0 and written.st_mtime >= reserved,
            'pytest wrote no JUnit XML over the reserved one: no parity case ran')
    return completed.returncode, argv


def collect(log, reviewed_commit, gpu=None, junit=None, out=None, allow_dirty=False,
            runner=subprocess.run):
    log = Path(log)
    junit = Path(junit) if junit else log.with_suffix('.junit.xml')
    out = Path(out) if out else log.with_suffix('.receipt.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', reviewed_commit + '^{commit}'],
                                     cwd=str(REPO), text=True).strip()
    require(commit == head, 'parity must run at the reviewed commit: ' + commit + ' != ' + head)
    state = p.checked_git_state(REPO, True, allow_dirty)
    device = cuda_device()
    status, argv = run(log, junit, gpu, runner)
    recorded = parsed_outcomes(junit)
    check(recorded, status)
    receipt = dict(schema_version=1, passed=True, tests=dict(recorded),
                   n_tests=len(TESTS), pytest_exit=status, git_head=head,
                   reviewed_commit=commit, git_state=state, cuda_device=device, gpu=gpu,
                   allow_dirty_used=bool(allow_dirty and state['dirty_outside_worklog']),
                   command=argv, python=sys.executable, environment=p.environment(),
                   log=dict(path=str(log.resolve()), sha256=p.sha256_file(log)),
                   junit=dict(path=str(junit.resolve()), sha256=p.sha256_file(junit)))
    p.write_manifest(out, receipt)
    print('wrote ' + str(out), flush=True)
    return receipt


def main(argv=None, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, help='parity log; created exclusively')
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--gpu', help='physical device for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--junit', help='default: the log path with a .junit.xml suffix')
    parser.add_argument('--out', help='default: the log path with a .receipt.json suffix')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='run outside a clean checkout; the record binder refuses such evidence')
    args = parser.parse_args(argv)
    return collect(args.log, args.reviewed_commit, args.gpu, args.junit, args.out,
                   args.allow_dirty, runner)


if __name__ == '__main__':
    try:
        print(json.dumps(main(), sort_keys=True, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
