# Runtime image for Milvus HIP standalone (ROCm).
# Same layout as official milvus GPU Docker: COPY bin/ configs/ lib/.
# Build context is the packed tree from pack_from_lab.sh (directory with bin/, configs/, lib/).
#
#   bash pack_from_lab.sh
#   docker build -f Dockerfile -t milvus-hip:v2.5.4-rocm context/
#
# Do not use milvusdb/milvus GPU tags on a Radeon — those are CUDA.
# Default offload arch for this tech preview is gfx1100. Instinct is untested.

ARG ROCM_IMAGE=rocm/dev-ubuntu-22.04:7.0.2
FROM ${ROCM_IMAGE}

ARG TARGETARCH=amd64

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates libaio-dev libgomp1 libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tini "https://github.com/krallin/tini/releases/download/v0.19.0/tini-${TARGETARCH}" && \
    chmod +x /tini

COPY ./bin/ /milvus/bin/
COPY ./configs/ /milvus/configs/
COPY ./lib/ /milvus/lib/
COPY ./docker-entrypoint.sh /milvus/docker-entrypoint.sh
RUN chmod 755 /milvus/bin/milvus /milvus/docker-entrypoint.sh && \
    find /milvus/lib -type f -exec chmod a+rX {} +

ENV PATH=/milvus/bin:/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/milvus/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0

ENTRYPOINT ["/milvus/docker-entrypoint.sh"]
CMD ["milvus", "run", "standalone"]

WORKDIR /milvus
