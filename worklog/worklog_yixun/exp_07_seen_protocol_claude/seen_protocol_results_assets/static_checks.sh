#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export XRIR_DATA_PATH="${XRIR_DATA_PATH:-/home/yixunhu/data_cache/AcousticRooms}"
record_assets=worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets
record_producers="tools/exp07_record.py tools/exp07_profiles.py tools/exp07_table.py tools/exp07_pairs.py
tools/exp07_train.py tools/exp07_provenance.py tools/exp07_launcher.py tools/exp07_probe.py
tools/exp07_gates.py tools/exp07_eval.py tools/exp07_eval_launch.py tools/exp07_audit.py
tools/exp07_manifests.py"
record_python=/home/yixunhu/miniconda3/envs/xRIR/bin/python
record_status=0
"$record_python" -c 'import py_compile,sys,tempfile; cache=tempfile.TemporaryDirectory(); [py_compile.compile(p,cfile=cache.name+"/"+str(i)+".pyc",doraise=True) for i,p in enumerate(sys.argv[1:])]' \
    "$record_assets"/{make_results_md,make_results_html,make_latex,bind_provenance,check_record}.py \
    $record_producers || record_status=1
git diff --check -- "$record_assets" $record_producers tools/exp07_launch.sh || record_status=1
bash -n tools/exp07_launch.sh || record_status=1
"$record_python" -m pytest tests/test_exp07_record_tools.py tests/test_exp07_table.py \
    tests/test_exp07_pairs.py tests/test_exp07_profiles.py -q -p no:cacheprovider || record_status=1
# Both collection orders: the record assets of exp_03, exp_04 and exp_07 share file names.
"$record_python" -m pytest tests/test_exp07_record_tools.py tests/test_exp04_record_tools.py \
    tests/test_exp03_record_tools.py -q -p no:cacheprovider || record_status=1
"$record_python" -m pytest tests/test_exp03_record_tools.py tests/test_exp04_record_tools.py \
    tests/test_exp07_record_tools.py -q -p no:cacheprovider || record_status=1
exit "$record_status"
