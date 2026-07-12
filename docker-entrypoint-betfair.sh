#!/bin/sh
set -e

# If a UK SOCKS proxy is configured (needed because Betfair blocks
# BETTING_RESTRICTED_LOCATION jurisdictions, e.g. a US-hosted VPS), open a
# self-healing SSH dynamic port forward to it before starting the poller.
# See docs/superpowers/specs/2026-07-11-live-betfair-odds-pipeline-design.md.
if [ -n "$UK_PROXY_HOST" ] && [ -n "$UK_PROXY_USER" ] && [ -f "$UK_PROXY_KEY_FILE" ]; then
    echo "Iniciando tunel SSH SOCKS5 hacia $UK_PROXY_USER@$UK_PROXY_HOST..."
    (
        while true; do
            ssh -N -D 1080 \
                -o StrictHostKeyChecking=accept-new \
                -o ServerAliveInterval=30 \
                -o ServerAliveCountMax=3 \
                -o ExitOnForwardFailure=yes \
                -i "$UK_PROXY_KEY_FILE" \
                "$UK_PROXY_USER@$UK_PROXY_HOST"
            echo "Tunel SSH caido, reintentando en 5s..."
            sleep 5
        done
    ) &
    sleep 3
fi

exec python -m goles.betfair.poller
