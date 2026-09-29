#!/bin/bash
# anomaly A/E run chain (after predictions 534805c). Logs to natural/anomalyA/.
set -x
D=/home/user/shape-zero/natural/anomalyA
WT=/tmp/claude-0/-home-user-shape-zero/7f4f19ab-d202-57e0-98c4-ef10d02be803/scratchpad/mainwt
cd $D && python3 anomE_purity.py > anomE_purity_output.txt 2>&1
cd $WT/shape_zero_tests && python3 certify_gates.py run > $D/certify_Aprime_run.log 2>&1
cd $WT/shape_zero_tests && python3 certify_gates.py evaluate > $D/certify_Aprime_rerun_output.txt 2>&1
cd $D && python3 anomA_smooth.py q1 > smooth_q1.log 2>&1
cd $D && python3 anomA_smooth.py q3 > smooth_q3.log 2>&1
echo CHAIN DONE > $D/chain_done.txt
