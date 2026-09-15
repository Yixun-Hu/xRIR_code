"""exp_06's alpha-aware paired bootstrap: exp_02's conventions, with configurable tails."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sim_to_real.summarize_haa import METRICS, ROOMS, load_runs, paired
from tools import exp06_bootstrap as subject

ROOT = Path(__file__).resolve().parents[1]
SIM2REAL = ROOT / 'ckpt/sim2real'
CANONICAL = SIM2REAL / 'stats.json'


def synthetic(n_queries=12, n_seeds=3, seed=4):
    """Paired rows of `n_queries` queries measured under `n_seeds` training seeds."""
    generator = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_queries), n_seeds)
    seeds = np.tile(np.arange(n_seeds), n_queries)
    a = generator.normal(1.0, 0.3, clusters.size)
    b = a - generator.normal(0.1, 0.2, clusters.size)
    return a, b, clusters, seeds


# --- exp_02's conventions, reproduced ---------------------------------------------------


def test_the_alpha_aware_bootstrap_is_the_pinned_one_at_alpha_0_05():
    a, b, clusters, seeds = synthetic()
    mean, lo, hi, _, frac, extra = paired(a, b, 10000, seed=0, clusters=clusters, seeds=seeds)
    got = subject.paired_intervals(a, b, clusters, seeds, alpha=0.05, n_boot=10000, seed=0)
    assert got['diff'] == mean and got['frac_a_lower'] == frac
    assert (got['query']['lo'], got['query']['hi']) == (lo, hi)
    assert (got['two_way']['lo'], got['two_way']['hi']) == extra['two_way']
    assert got['n_boot'] == 10000 and got['seed'] == 0 and got['alpha'] == 0.05
    assert got['tails'] == [2.5, 97.5]
    assert got['n_clusters'] == 12 and got['n_seeds'] == 3 and got['n'] == 36


def canonical_cells():
    if not CANONICAL.is_file():
        return []
    return [(cell['room'], cell['metric']) for cell in json.loads(CANONICAL.read_text())['paired']]


def cell_arrays(runs, room, key):
    """Exactly summarize_haa's pooling: ordered seeds, concatenated rows, finite mask."""
    cyl = {os.path.basename(r['dir']): r for r in runs[('cyl', 'fine-tuned')]}
    ctl = {os.path.basename(r['dir']): r for r in runs[('control', 'fine-tuned')]}
    a, b, cl, sd = [], [], [], []
    for s in sorted(set(cyl) & set(ctl)):
        A, B = cyl[s]['per'][room], ctl[s]['per'][room]
        x, y = np.asarray(A[key], float), np.asarray(B[key], float)
        a.append(x)
        b.append(y)
        cl.append(np.asarray(A['index']))
        sd.append(np.full(len(x), s))
    a, b, cl, sd = [np.concatenate(v) for v in (a, b, cl, sd)]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], cl[ok], sd[ok]


@pytest.mark.skipif(not CANONICAL.is_file(), reason='needs the exp_02 results')
@pytest.mark.parametrize('room,key', canonical_cells())
def test_every_canonical_exp02_cell_reproduces_exactly(room, key):
    stats = json.loads(CANONICAL.read_text())
    cell = next(c for c in stats['paired'] if c['room'] == room and c['metric'] == key)
    a, b, clusters, seeds = cell_arrays(load_runs(str(SIM2REAL)), room, key)
    assert len(a) == cell['n_valid']
    got = subject.paired_intervals(a, b, clusters, seeds, alpha=0.05,
                                   n_boot=stats['n_boot'], seed=stats['boot_seed'])
    assert abs(got['diff'] - cell['diff']) == 0.0
    assert abs(got['query']['lo'] - cell['lo']) == 0.0
    assert abs(got['query']['hi'] - cell['hi']) == 0.0
    assert abs(got['two_way']['lo'] - cell['lo_two_way']) == 0.0
    assert abs(got['two_way']['hi'] - cell['hi_two_way']) == 0.0
    assert got['n_clusters'] == cell['n_queries'] and got['n_seeds'] == cell['n_seeds']


@pytest.mark.skipif(not CANONICAL.is_file(), reason='needs the exp_02 results')
def test_the_canonical_cells_are_the_eleven_the_protocol_requires():
    cells = canonical_cells()
    assert len(cells) == 11
    assert {room for room, _ in cells} == set(ROOMS)
    assert {key for _, key in cells} <= {key for key, _, _ in METRICS}


# --- the adjusted tails -----------------------------------------------------------------


def test_the_adjusted_tails_are_percentiles_of_the_very_same_samples():
    a, b, clusters, seeds = synthetic()
    alpha = subject.bonferroni_alpha(0.05, 11)
    nominal = subject.paired_intervals(a, b, clusters, seeds, alpha=0.05, n_boot=2000,
                                       seed=3, return_samples=True)
    adjusted = subject.paired_intervals(a, b, clusters, seeds, alpha=alpha, n_boot=2000,
                                        seed=3, return_samples=True)
    for scheme in ('query', 'two_way'):
        samples = adjusted['samples'][scheme]
        assert np.array_equal(samples, nominal['samples'][scheme])
        low, high = np.nanpercentile(samples, [100 * 0.05 / 22, 100 * (1 - 0.05 / 22)])
        assert adjusted[scheme]['lo'] == low and adjusted[scheme]['hi'] == high
        assert adjusted[scheme]['lo'] <= nominal[scheme]['lo']
        assert adjusted[scheme]['hi'] >= nominal[scheme]['hi']
    assert adjusted['tails'] == [100 * 0.05 / 22, 100 * (1 - 0.05 / 22)]


def test_the_bootstrap_summary_describes_the_samples_it_used():
    a, b, clusters, seeds = synthetic()
    got = subject.paired_intervals(a, b, clusters, seeds, n_boot=500, seed=1,
                                   return_samples=True)
    for scheme in ('query', 'two_way'):
        samples = got['samples'][scheme]
        assert len(samples) == 500 and got[scheme]['n_nan'] == int(np.isnan(samples).sum())
        assert got[scheme]['mean'] == float(np.nanmean(samples))
        assert got[scheme]['sd'] == float(np.nanstd(samples))


@pytest.mark.parametrize('alpha,m,expected', [(0.05, 11, 0.05 / 11), (0.05, 1, 0.05),
                                              (0.1, 4, 0.025)])
def test_bonferroni_alpha_divides_the_family(alpha, m, expected):
    assert subject.bonferroni_alpha(alpha, m) == expected


@pytest.mark.parametrize('alpha,m', [(0.0, 3), (1.0, 3), (-0.1, 3), (0.05, 0),
                                     (0.05, 1.5), (0.05, True), (float('nan'), 3)])
def test_bonferroni_alpha_refuses_a_malformed_family(alpha, m):
    with pytest.raises(ValueError):
        subject.bonferroni_alpha(alpha, m)


@pytest.mark.parametrize('case', ['nan_a', 'nan_b', 'length', 'empty', 'clusters',
                                  'seeds', 'alpha', 'n_boot', 'seed'])
def test_degenerate_inputs_are_refused(case):
    a, b, clusters, seeds = synthetic(n_queries=4)
    kwargs = {}
    if case == 'nan_a':
        a = a.copy()
        a[0] = np.nan
    elif case == 'nan_b':
        b = b.copy()
        b[1] = np.inf
    elif case == 'length':
        b = b[:-1]
    elif case == 'empty':
        a, b, clusters, seeds = a[:0], b[:0], clusters[:0], seeds[:0]
    elif case == 'clusters':
        clusters = clusters[:-1]
    elif case == 'seeds':
        seeds = seeds[:-1]
    elif case == 'alpha':
        kwargs['alpha'] = 0.0
    elif case == 'n_boot':
        kwargs['n_boot'] = 0
    else:
        kwargs['seed'] = -1
    with pytest.raises(ValueError):
        subject.paired_intervals(a, b, clusters, seeds, n_boot=50, **kwargs)


# --- convergence ------------------------------------------------------------------------


def test_identical_intervals_converge_with_a_zero_ratio():
    got = subject.convergence_endpoints(lambda seed: (-1.0, 1.0))
    assert got['passed'] is True and got['ratio'] == 0.0 and got['movement'] == 0.0
    assert got['width'] == 2.0 and got['tolerance'] == 0.10
    assert got['seed_a'] == {'seed': 0, 'lo': -1.0, 'hi': 1.0}
    assert got['seed_b'] == {'seed': 1, 'lo': -1.0, 'hi': 1.0}


@pytest.mark.parametrize('shift,passed', [(0.1, True), (0.2, True), (0.3, False),
                                          (0.21, False)])
def test_a_shifted_interval_of_equal_width_fails_beyond_the_tolerance(shift, passed):
    """Width 2, so the tolerance admits endpoint movement up to 0.2."""
    def interval(seed):
        return (-1.0 + shift * seed, 1.0 + shift * seed)

    got = subject.convergence_endpoints(interval)
    assert got['passed'] is passed
    assert got['movement'] == pytest.approx(shift) and got['ratio'] == pytest.approx(shift / 2)


@pytest.mark.parametrize('interval', [(1.0, 1.0), (0.0, 0.0)])
def test_a_zero_width_interval_is_refused(interval):
    with pytest.raises(ValueError):
        subject.convergence_endpoints(lambda seed: interval)


@pytest.mark.parametrize('bad', [(1.0, -1.0), (np.nan, 1.0), (1.0, np.inf), (1.0,),
                                 'not an interval'])
def test_a_malformed_interval_is_refused(bad):
    with pytest.raises(ValueError):
        subject.convergence_endpoints(lambda seed: (-1.0, 1.0) if seed == 0 else bad)


def widths(sequence):
    """A factory whose intervals stop moving once n_boot reaches the given size."""
    def factory(n_boot):
        movement = sequence.get(n_boot, 1.0)
        return lambda seed: (-1.0 + movement * seed, 1.0 + movement * seed)
    return factory


def test_a_converged_interval_is_returned_at_the_first_size():
    got = subject.converged_interval(widths({1000: 0.0}), n_boot=1000)
    assert got['status'] == 'converged' and got['n_boot'] == 1000
    assert got['interval'] == (-1.0, 1.0) and len(got['attempts']) == 1


def test_a_failure_quadruples_the_sample_count_once():
    got = subject.converged_interval(widths({4000: 0.0}), n_boot=1000)
    assert got['status'] == 'converged' and got['n_boot'] == 4000
    assert [attempt['n_boot'] for attempt in got['attempts']] == [1000, 4000]
    assert got['attempts'][0]['passed'] is False


def test_a_verdict_is_withheld_when_it_still_does_not_converge():
    got = subject.converged_interval(widths({}), n_boot=1000)
    assert got['status'] == 'not_converged' and got['interval'] is None
    assert [attempt['n_boot'] for attempt in got['attempts']] == [1000, 4000]


@pytest.mark.parametrize('n_boot', [0, -1, 1.5, True])
def test_converged_interval_refuses_a_malformed_sample_count(n_boot):
    with pytest.raises(ValueError):
        subject.converged_interval(widths({}), n_boot=n_boot)
