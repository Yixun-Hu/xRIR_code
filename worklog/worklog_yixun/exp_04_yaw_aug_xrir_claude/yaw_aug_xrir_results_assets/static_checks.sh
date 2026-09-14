#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/xrir_round8_numba
record_assets=worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets
record_python=/home/yixunhu/miniconda3/envs/xRIR/bin/python
record_status=0
"$record_python" -c 'import glob,py_compile,sys,tempfile; cache=tempfile.TemporaryDirectory(); [py_compile.compile(p,cfile=cache.name+"/"+str(i)+".pyc",doraise=True) for i,p in enumerate(sys.argv[1:]+glob.glob("tools/exp04_descriptive.py"))]' "$record_assets"/{make_results_md,make_results_html,bind_provenance,check_record}.py tools/{exp04_record,exp04_profiles,paired_compare,results_table}.py || record_status=1
git diff --check -- "$record_assets" tools/{exp04_record,exp04_profiles,exp04_descriptive,paired_compare,results_table}.py || record_status=1
"$record_python" -m pytest tests/test_exp04_record_tools.py -k refusals -q -p no:cacheprovider || record_status=1
exit "$record_status"
