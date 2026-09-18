#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export XRIR_DATA_PATH="${XRIR_DATA_PATH:-/home/yixunhu/data_cache/AcousticRooms}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/xrir_exp05_numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/xrir_exp05_mpl}"
record_assets=worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets
record_sources="tools/exp05_record.py tools/exp05_profiles.py tools/exp05_params.py
tools/param_curve.py tests/exp05_record_fixture.py tests/test_exp05_record_tools.py"
record_python=/home/yixunhu/miniconda3/envs/xRIR/bin/python
record_status=0
"$record_python" -c 'import py_compile,sys,tempfile; cache=tempfile.TemporaryDirectory(); [py_compile.compile(p,cfile=cache.name+"/"+str(i)+".pyc",doraise=True) for i,p in enumerate(sys.argv[1:])]' \
    "$record_assets"/{make_results_md,make_results_html,make_figures,bind_provenance,check_record}.py \
    $record_sources || record_status=1
git diff --check -- "$record_assets" $record_sources || record_status=1
bash -n "$record_assets/static_checks.sh" || record_status=1
"$record_python" -m pytest tests/test_exp05_record_tools.py -q -p no:cacheprovider || record_status=1
# Both collection orders: the record assets of exp_03, exp_04, exp_05 and exp_07 share file names.
"$record_python" -m pytest tests/test_exp05_record_tools.py tests/test_exp04_record_tools.py \
    tests/test_exp03_record_tools.py tests/test_exp07_record_tools.py -q -p no:cacheprovider \
    || record_status=1
"$record_python" -m pytest tests/test_exp07_record_tools.py tests/test_exp03_record_tools.py \
    tests/test_exp04_record_tools.py tests/test_exp05_record_tools.py -q -p no:cacheprovider \
    || record_status=1
exit "$record_status"
