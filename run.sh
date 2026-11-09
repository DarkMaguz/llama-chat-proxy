#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ".venv/bin/activate"

printf "Enter key: "
read -rs DB_KEY
echo ""
if [[ -z $DB_KEY ]]; then
    echo "DB_KEY required" >&2
    exit 1
fi

if [ "${#DB_KEY}" -ne 128 ]; then
    echo "Bad key!" >&2
    exit 1
fi

export LLAMA_PROXY_KEY=$DB_KEY

exec python -m llama_chat_proxy
