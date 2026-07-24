#!/usr/bin/env bash

set -euo pipefail

action=${1:?usage: milvus_service.sh <start|stop|status>}
health_uri=${MILVUS_CI_HEALTH_URI:-http://127.0.0.1:9091/healthz}

start_dependencies() {
    local containers=${MILVUS_CI_DEPENDENCY_CONTAINERS:-}
    local container
    IFS=',' read -ra names <<< "${containers}"
    for container in "${names[@]}"; do
        if [[ -n "${container}" ]]; then
            docker start "${container}" >/dev/null
        fi
    done
}

wait_healthy() {
    local attempts=${MILVUS_CI_HEALTH_ATTEMPTS:-90}
    local delay=${MILVUS_CI_HEALTH_DELAY_SECONDS:-2}
    local attempt
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl -fsS "${health_uri}" >/dev/null 2>&1; then
            echo "Milvus is healthy after $((attempt * delay))s"
            return 0
        fi
        sleep "${delay}"
    done
    echo "Milvus did not become healthy: ${health_uri}" >&2
    return 1
}

stop_milvus() {
    if [[ -n "${MILVUS_CI_CONTAINER:-}" ]]; then
        if [[ "$(docker inspect -f '{{.State.Running}}' "${MILVUS_CI_CONTAINER}" 2>/dev/null || true)" == "true" ]]; then
            docker stop --time 30 "${MILVUS_CI_CONTAINER}" >/dev/null
        fi
    fi

    local pids=()
    if [[ -n "${MILVUS_CI_PID_FILE:-}" && -s "${MILVUS_CI_PID_FILE}" ]]; then
        pids+=("$(<"${MILVUS_CI_PID_FILE}")")
        rm -f "${MILVUS_CI_PID_FILE}"
    fi
    if [[ -n "${MILVUS_CI_BINARY:-}" || -n "${MILVUS_CI_ROOT:-}" ]]; then
        local binary
        binary=$(realpath "${MILVUS_CI_BINARY:-${MILVUS_CI_ROOT}/bin/milvus}")
        local candidate
        for candidate in $(pgrep -x milvus || true); do
            if [[ "$(readlink -f "/proc/${candidate}/exe" 2>/dev/null || true)" == "${binary}" ]]; then
                pids+=("${candidate}")
            fi
        done
    fi

    local pid
    for pid in $(printf '%s\n' "${pids[@]}" | sort -u); do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            for _ in {1..30}; do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "${pid}" 2>/dev/null; then
                kill -KILL "${pid}"
            fi
        fi
    done

    if curl -fsS "${health_uri}" >/dev/null 2>&1; then
        echo "Milvus is still healthy after stop" >&2
        return 1
    fi
    echo "Milvus stopped"
}

start_binary() {
    local root=${MILVUS_CI_ROOT:?MILVUS_CI_ROOT is required for a host binary}
    local binary=${MILVUS_CI_BINARY:-${root}/bin/milvus}
    local pid_file=${MILVUS_CI_PID_FILE:?MILVUS_CI_PID_FILE is required}
    local log_file=${MILVUS_CI_LOG_FILE:?MILVUS_CI_LOG_FILE is required}

    mkdir -p "$(dirname "${pid_file}")" "$(dirname "${log_file}")"
    (
        cd "${root}"
        # shellcheck disable=SC1091
        set +u
        source scripts/setenv.sh >/dev/null
        set -u
        export LD_LIBRARY_PATH="${MILVUS_CI_EXTRA_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
        unset RUNNER_TRACKING_ID
        setsid --fork env \
            ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
            HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
            "${binary}" run standalone >"${log_file}" 2>&1 </dev/null
        for _ in {1..50}; do
            local candidate
            for candidate in $(pgrep -x milvus || true); do
                if [[ "$(readlink -f "/proc/${candidate}/exe" 2>/dev/null || true)" == "$(realpath "${binary}")" ]]; then
                    echo "${candidate}" >"${pid_file}"
                    return 0
                fi
            done
            sleep 0.1
        done
        echo "Milvus process was not created" >&2
        return 1
    )
}

start_milvus() {
    if curl -fsS "${health_uri}" >/dev/null 2>&1; then
        echo "Milvus is already healthy"
        return 0
    fi
    start_dependencies
    if [[ -n "${MILVUS_CI_CONTAINER:-}" ]]; then
        docker start "${MILVUS_CI_CONTAINER}" >/dev/null
    else
        start_binary
    fi
    wait_healthy
}

case "${action}" in
    start)
        start_milvus
        ;;
    stop)
        stop_milvus
        ;;
    status)
        curl -fsS "${health_uri}"
        ;;
    *)
        echo "unknown action: ${action}" >&2
        exit 2
        ;;
esac
