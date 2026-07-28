#!/usr/bin/env bash

set -euo pipefail

action=${1:?usage: milvus_service.sh <start|stop|status>}
grpc_port=${MILVUS_CI_GRPC_PORT:-19530}
internal_grpc_port=${MILVUS_CI_INTERNAL_GRPC_PORT:-19529}
metrics_port=${MILVUS_CI_METRICS_PORT:-9091}
health_uri=${MILVUS_CI_HEALTH_URI:-http://127.0.0.1:${metrics_port}/healthz}
runtime_dir=${MILVUS_CI_RUNTIME_DIR:?MILVUS_CI_RUNTIME_DIR is required}

runtime_env=(
    "MILVUS_CI_MANAGED=1"
    "PROXY_PORT=${grpc_port}"
    "PROXY_INTERNAL_PORT=${internal_grpc_port}"
    "METRICS_PORT=${metrics_port}"
    "ETCD_ENDPOINTS=${MILVUS_CI_ETCD_ENDPOINTS:-localhost:2379}"
    "ETCD_ROOT_PATH=${MILVUS_CI_ETCD_ROOT_PATH:?MILVUS_CI_ETCD_ROOT_PATH is required}"
    "MINIO_ADDRESS=${MILVUS_CI_MINIO_ADDRESS:-localhost:9000}"
    "MINIO_ROOT_PATH=${MILVUS_CI_MINIO_ROOT_PATH:?MILVUS_CI_MINIO_ROOT_PATH is required}"
)

is_container_mode() {
    [[ -n "${MILVUS_CI_IMAGE:-}" ]]
}

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

stop_container() {
    local containers=()
    mapfile -t containers < <(
        docker ps --all --quiet --filter label=com.dxc.milvus.gpu-ci=true
    )
    local container
    for container in "${containers[@]}"; do
        if [[ -n "${MILVUS_CI_LOG_FILE:-}" ]]; then
            mkdir -p "$(dirname "${MILVUS_CI_LOG_FILE}")"
            docker logs "${container}" >>"${MILVUS_CI_LOG_FILE}" 2>&1 || true
        fi
        docker rm --force "${container}" >/dev/null
    done
}

stop_binary() {
    local binary=${MILVUS_CI_BINARY:?MILVUS_CI_BINARY is required}
    local pid_file=${MILVUS_CI_PID_FILE:?MILVUS_CI_PID_FILE is required}
    local pids=()

    if [[ -s "${pid_file}" ]]; then
        pids+=("$(<"${pid_file}")")
        rm -f "${pid_file}"
    fi

    if [[ -e "${binary}" ]]; then
        local candidate
        for candidate in $(pgrep -x milvus || true); do
            if [[ "$(readlink -f "/proc/${candidate}/exe" 2>/dev/null || true)" == "$(realpath "${binary}")" ]]; then
                pids+=("${candidate}")
            fi
        done
    fi
    local candidate
    for candidate in $(pgrep -x milvus || true); do
        if tr '\0' '\n' <"/proc/${candidate}/environ" 2>/dev/null |
            grep -qx 'MILVUS_CI_MANAGED=1'; then
            pids+=("${candidate}")
        fi
    done

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
}

stop_milvus() {
    if is_container_mode; then
        stop_container
    else
        stop_binary
    fi

    if curl -fsS "${health_uri}" >/dev/null 2>&1; then
        echo "Another Milvus process still owns ${health_uri}" >&2
        return 1
    fi
    echo "Milvus stopped"
}

start_binary() {
    local root=${MILVUS_CI_ROOT:?MILVUS_CI_ROOT is required}
    local binary=${MILVUS_CI_BINARY:?MILVUS_CI_BINARY is required}
    local pid_file=${MILVUS_CI_PID_FILE:?MILVUS_CI_PID_FILE is required}
    local log_file=${MILVUS_CI_LOG_FILE:?MILVUS_CI_LOG_FILE is required}

    if [[ ! -x "${binary}" ]]; then
        echo "Exact-SHA Milvus binary is missing: ${binary}" >&2
        return 1
    fi

    mkdir -p "${runtime_dir}" "$(dirname "${pid_file}")" "$(dirname "${log_file}")"
    (
        cd "${root}"
        set +u
        # shellcheck disable=SC1091
        source scripts/setenv.sh >/dev/null
        set -euo pipefail
        export LD_LIBRARY_PATH="${MILVUS_CI_EXTRA_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
        unset RUNNER_TRACKING_ID
        setsid --fork env \
            "${runtime_env[@]}" \
            MILVUSCONF="${root}/configs" \
            LOCAL_STORAGE_PATH="${runtime_dir}/data" \
            ROCKSMQ_PATH="${runtime_dir}/rdb_data" \
            NATSMQ_SERVER_STOREDIR="${runtime_dir}/nats" \
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
        echo "Exact-SHA Milvus process was not created" >&2
        return 1
    )
}

start_container() {
    local image=${MILVUS_CI_IMAGE:?MILVUS_CI_IMAGE is required}
    local container=${MILVUS_CI_CONTAINER:?MILVUS_CI_CONTAINER is required}
    local network=${MILVUS_CI_DOCKER_NETWORK:?MILVUS_CI_DOCKER_NETWORK is required}

    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        echo "Exact-SHA Milvus image is missing: ${image}" >&2
        return 1
    fi
    local image_commit
    image_commit=$(
        docker image inspect \
            --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
            "${image}"
    )
    if [[ -n "${GITHUB_SHA:-}" && "${image_commit}" != "${GITHUB_SHA}" ]]; then
        echo "Milvus image revision ${image_commit} does not match ${GITHUB_SHA}" >&2
        return 1
    fi

    mkdir -p "${runtime_dir}"
    stop_container
    local docker_env=()
    local item
    for item in "${runtime_env[@]}"; do
        docker_env+=(--env "${item}")
    done
    docker run --detach \
        --name "${container}" \
        --label com.dxc.milvus.gpu-ci=true \
        --label "org.opencontainers.image.revision=${GITHUB_SHA:-unknown}" \
        --network "${network}" \
        --gpus all \
        --ipc host \
        --publish "127.0.0.1:${grpc_port}:${grpc_port}" \
        --publish "127.0.0.1:${metrics_port}:${metrics_port}" \
        --volume "${runtime_dir}:/var/lib/milvus" \
        "${docker_env[@]}" \
        --env LOCAL_STORAGE_PATH=/var/lib/milvus/data \
        --env ROCKSMQ_PATH=/var/lib/milvus/rdb_data \
        --env NATSMQ_SERVER_STOREDIR=/var/lib/milvus/nats \
        "${image}" \
        milvus run standalone >/dev/null
}

verify_server_commit() {
    local expected=${GITHUB_SHA:-}
    [[ -z "${expected}" ]] && return 0

    local metrics_uri=${MILVUS_METRICS_URI:-http://127.0.0.1:${metrics_port}/metrics}
    local server_commit
    server_commit=$(
        curl -fsS "${metrics_uri}" |
            sed -n 's/^milvus_build_info{.*git_commit="\([^"]*\)".*/\1/p' |
            head -n 1
    )
    if [[ -z "${server_commit}" || "${expected}" != "${server_commit}"* ]]; then
        echo "Running Milvus commit ${server_commit:-missing} does not match ${expected}" >&2
        return 1
    fi
    echo "Running Milvus commit verified: ${server_commit}"
}

start_milvus() {
    if curl -fsS "${health_uri}" >/dev/null 2>&1; then
        echo "Port is already served before exact-SHA Milvus start" >&2
        return 1
    fi
    start_dependencies
    if is_container_mode; then
        start_container
    else
        start_binary
    fi
    wait_healthy
    verify_server_commit
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
