#!/bin/bash
# exp_03 static checks (Planner): every command echoed with its exit status; overall exit non-zero if any fails.
cd /home/yixunhu/codespace/xRIR_code || exit 2
export PYTHONPATH="${PYTHONPATH:-}:/home/yixunhu/codespace/xRIR_code"
P="$HOME/miniconda3/envs/xRIR/bin/python"; A=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets; E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude
rc_all=0
run() { echo "\$ $*"; "$@"; local rc=$?; echo "  -> exit $rc"; [ $rc -eq 0 ] || rc_all=1; }
echo "HEAD $(git rev-parse HEAD)"
run "$P" -m py_compile "$A/bind_provenance.py" "$A/check_sweep_acceptance.py" "$A/make_results_html.py" "$A/make_results_md.py" "$A/eval_stub.py" tools/summarize_yaw.py tests/test_exp03_record_tools.py
run bash -n "$A/launch_yaw_sweep_v3.sh"
run bash -n "$A/launch_yaw_sweep_v2.sh"
run git diff --check
echo "\$ trailing-whitespace scan of the untracked record files (grep -lP '[ \\t]+\$')"
hits=$(for f in $(git ls-files --others --exclude-standard "$E" tests); do grep -lP '[ \t]+$' "$f"; done); if [ -n "$hits" ]; then echo "$hits"; echo "  -> exit 1"; rc_all=1; else echo "  -> exit 0 (no hits)"; fi
echo "STATIC_CHECKS $([ $rc_all -eq 0 ] && echo PASS || echo FAIL)"; exit $rc_all
