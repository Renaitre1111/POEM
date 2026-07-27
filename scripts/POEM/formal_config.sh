#!/usr/bin/env bash

poem_config_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
poem_formal_jobs="$poem_config_dir/formal_jobs.tsv"

formal_job_table() {
    awk 'NF && $1 !~ /^#/' "$poem_formal_jobs"
}

formal_job_for() {
    local dataset=$1 pred_len=$2
    formal_job_table | awk -v d="$dataset" -v p="$pred_len" \
        '$2 == d && $5 == p { print; exit }'
}

enc_in_for() {
    case "$1" in
        ETTh1|ETTh2|ETTm1|ETTm2) echo 7 ;;
        weather) echo 21 ;;
        electricity) echo 321 ;;
        traffic) echo 862 ;;
    esac
}
