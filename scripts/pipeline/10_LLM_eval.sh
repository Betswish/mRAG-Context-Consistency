#!/bin/bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

cd_repo_root
sample_script="$REPO_ROOT/sample_instances.py"

if [[ ! -f "$sample_script" ]]; then
    echo "Missing sample_instances.py in the repository root."
    exit 1
fi

for dataset in XQUAD_open
do
    for lang in en ar de el es hi ro ru th tr vi zh
    # for lang in ar el es hi ro ru th tr vi zh
    do
        echo "$lang"
        python "$sample_script" --lang "$lang" --dataset "$dataset" --num 50
    done
done
