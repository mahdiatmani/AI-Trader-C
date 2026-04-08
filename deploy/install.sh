#!/usr/bin/env bash
# One-shot installer for the GA trading bot on a fresh Ubuntu box.
#
# Usage (as root or with sudo):
#   sudo bash deploy/install.sh
#
# Idempotent — safe to re-run after a `git pull`.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/ai-trader}"
SERVICE_USER="${SERVICE_USER:-trader}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Installing GA trading bot to ${REPO_DIR} as user ${SERVICE_USER}"

# 1. system deps
apt-get update -y
apt-get install -y --no-install-recommends \
    "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev" \
    build-essential git ca-certificates

# 2. service user
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

# 3. fix ownership of the repo
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${REPO_DIR}"

# 4. virtualenv + dependencies
sudo -u "${SERVICE_USER}" bash -c "
    set -e
    cd '${REPO_DIR}'
    if [ ! -d .venv ]; then
        ${PYTHON_BIN} -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip wheel
    .venv/bin/pip install -r requirements.txt
    # Live trading needs MetaApi too:
    .venv/bin/pip install metaapi-cloud-sdk || true
"

# 5. environment file template (you fill in the secrets)
ENV_FILE=/etc/ai-trader.env
if [ ! -f "${ENV_FILE}" ]; then
    cat >"${ENV_FILE}" <<'ENVEOF'
# /etc/ai-trader.env — secrets for the GA trading bot service.
# This file is read by systemd via EnvironmentFile= and is NOT in git.
METAAPI_TOKEN=replace_me
METAAPI_ACCOUNT_ID=replace_me
ENVEOF
    chmod 600 "${ENV_FILE}"
    chown root:"${SERVICE_USER}" "${ENV_FILE}"
    echo "==> Created ${ENV_FILE} — edit it and put your real MetaApi credentials in."
fi

# 6. install the systemd unit
install -m 0644 "${REPO_DIR}/deploy/ai-trader.service" /etc/systemd/system/ai-trader.service
systemctl daemon-reload

cat <<MSG

Install complete.

Next steps:
    1. Edit ${ENV_FILE} and put your MetaApi credentials in.
    2. Drop your training CSV in ${REPO_DIR}/data/ and run training once:
         sudo -u ${SERVICE_USER} ${REPO_DIR}/.venv/bin/python ${REPO_DIR}/train_ga.py --csv XAUUSD_M5.csv
    3. Enable + start the service:
         sudo systemctl enable --now ai-trader.service
    4. Watch it run:
         journalctl -u ai-trader -f
         cat ${REPO_DIR}/logs/heartbeat.json
MSG
