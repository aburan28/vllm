# MooncakeConnector Usage Guide

## About Mooncake

Mooncake aims to enhance the inference efficiency of large language models (LLMs), especially in slow object storage environments, by constructing a multi-level caching pool on high-speed interconnected DRAM/SSD resources. Compared to traditional caching systems, Mooncake utilizes (GPUDirect) RDMA technology to transfer data directly in a zero-copy manner, while maximizing the use of multi-NIC resources on a single machine.

For more details about Mooncake, please refer to [Mooncake project](https://github.com/kvcache-ai/Mooncake) and [Mooncake documents](https://kvcache-ai.github.io/Mooncake/).

## Prerequisites

### Installation

Install mooncake through pip: `uv pip install mooncake-transfer-engine-cuda13`.

vLLM defaults to CUDA 13. On a CUDA 12 environment install `mooncake-transfer-engine` instead — the two are the same release built against different CUDA majors, and the wrong one fails to import with `libcudart.so.<major>: cannot open shared object file`.

Refer to [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions

## Usage

### Prefiller Node (192.168.0.2)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

### Decoder Node (192.168.0.3)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### Proxy

```bash
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

Now you can send requests to the proxy server through port 8000.

## Environment Variables

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`: Port for Mooncake bootstrap server
    - Default: 8998
    - Required only for prefiller instances
    - For headless instances, must be the same as the master instance
    - Each instance needs a unique port on its host; using the same port number across different hosts is fine

- `WITH_NVIDIA_PEERMEM`: Selects how mooncake registers GPU memory for RDMA. Read by mooncake, not vLLM.
    - Default: 1, which uses `ibv_reg_mr()` and requires the `nvidia-peermem` kernel module to be loaded
    - Set to 0 to use the DMA-BUF path, which does not need that module. Required on hosts where `nvidia-peermem` is not loaded, such as GB200
    - With the container image, pass it at run time: `docker run -e WITH_NVIDIA_PEERMEM=0 ...`
    - Symptom when left unset on such a host: `Failed to register memory <addr>: Bad address [14]` from `rdma_context.cpp`, and KV transfers fail

- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`: Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request. (Optional)
    - Default: 480
    - If a request is aborted and the decoder has not yet notified the prefiller, the prefill instance will release its KV-cache blocks after this timeout to avoid holding them indefinitely.

## KV Transfer Config

### KV Role Options

- **kv_producer**: For prefiller instances that generate KV caches
- **kv_consumer**: For decoder instances that consume KV caches from prefiller
- **kv_both**: Enables symmetric functionality where the connector can act as both producer and consumer. This provides flexibility for experimental setups and scenarios where the role distinction is not predetermined.

### kv_connector_extra_config

- **num_workers**: Size of thread pool for one prefiller worker to transfer KV caches by mooncake, or the number of batches kept in flight with `mooncake_async_transfer`. (default 10)
- **mooncake_protocol**: Mooncake connector protocol. (default "rdma")
- **device_name**: Comma-separated whitelist of RDMA devices (e.g. `"mlx5_0,mlx5_1"`) to restrict topology discovery to. Empty discovers every device. Useful on hosts exposing a mix of InfiniBand and RoCE ports, where both peers must settle on the same link layer.
- **mooncake_env**: Dict of Mooncake environment variables exported by every worker before its Transfer Engine is created. See [Non-RDMA transports](#non-rdma-transports-tcp-io_uring-zero-copy-dpdk). (default `{}`)
- **mooncake_async_transfer**: Submit KV transfers with the asynchronous batch API instead of blocking thread-pool calls. See [Non-RDMA transports](#non-rdma-transports-tcp-io_uring-zero-copy-dpdk). (default `true`)

## Non-RDMA transports (TCP, io_uring, zero copy, DPDK)

With an RDMA NIC Mooncake moves KV blocks with one-sided (GPUDirect) RDMA and none of this section applies. On plain Ethernet (cloud VMs without EFA or RoCE, virtio or gve NICs, hosts without RDMA configured) the connector has to use the Transfer Engine `tcp` transport instead, and the Mooncake knobs below decide how that transport moves the bytes.

### Selecting the TCP transport

Set `mooncake_protocol` to `"tcp"` on both the prefiller and the decoder; the two sides must use the same protocol. `MC_FORCE_TCP=1` additionally stops Mooncake from installing an RDMA transport that it might discover but that cannot reach the peer:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --kv-transfer-config \
  '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp","mooncake_env":{"MC_FORCE_TCP":"1","MC_TCP_IO_BACKEND":"io_uring"}}}'
```

### mooncake_env

Mooncake reads its `MC_*` settings from the process environment when the Transfer Engine is constructed and initialized. `mooncake_env` is a dict of variable name to value (strings, or ints that are converted to strings) that each worker applies with `os.environ.setdefault` right before creating its engine, so a single `--kv-transfer-config` configures the transport on every rank, including under Ray and multi-node launches where exporting variables by hand is error-prone. A variable that is already set in the worker's environment keeps its value. The applied variables are logged at startup.

### mooncake_async_transfer

By default the prefiller submits each batch of block copies with `batch_transfer_async_write` on its sender event loop and polls it there with the non-blocking `batch_transfer_poll` (sleeping 50 µs, doubling to 2 ms), falling back to `get_batch_transfer_status` on one poller thread with wheels that lack it, keeping at most `num_workers` batches in flight. This replaces `num_workers` threads that each blocked inside `batch_transfer_sync_write`. Transfer statistics (`MooncakeKVConnectorStats`) are recorded the same way in both modes, measured from submission to completion.

Set `mooncake_async_transfer` to `false` to keep the thread-pool path. The connector also falls back to it automatically when the installed Mooncake wheel does not expose the asynchronous batch API. The asynchronous engine API does not retry a batch that fails in flight, while `batch_transfer_sync_write` retries once per RDMA context, so multi-NIC RDMA deployments that rely on those retries may prefer the thread-pool path.

### Mooncake environment knobs

These variables are read by Mooncake, not vLLM, and are most conveniently passed through `mooncake_env`. Knobs marked "planned" belong to Mooncake's io_uring, zero-copy and DPDK work and only take effect once the installed wheel ships them; `mooncake.engine.SUPPORT_IOURING_TCP` and `mooncake.engine.SUPPORT_DPDK` report whether it does. Mooncake ignores variables it does not know.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MC_FORCE_TCP` | unset | Install only the `tcp` transport, even when RDMA devices are present |
| `MC_TRANSFER_TIMEOUT` | `30` | Seconds a batch may take before the engine reports it failed (plus 1 s per GiB) |
| `MC_TCP_SLICE_SIZE` | `65536` | Bytes per TCP chunk on the asio data plane |
| `MC_TCP_LANES_PER_PEER` | `4` | Pooled TCP connections per peer |
| `MC_TCP_IO_BACKEND` (planned) | `asio` | `asio` or `io_uring` data plane for the `tcp` transport, wire-compatible with each other |
| `MC_TCP_URING_WORKERS` (planned) | `min(4, ncpu/8)` | io_uring ring worker threads |
| `MC_TCP_URING_SQPOLL` (planned) | `0` | Enable `IORING_SETUP_SQPOLL` (spends one core to avoid submission syscalls) |
| `MC_TCP_URING_ZC_THRESHOLD` (planned) | `16384` | Minimum payload size sent with `SEND_ZC` |
| `MC_TCP_STAGING_CHUNK` (planned) | `2097152` | Pinned staging chunk size for device memory |
| `MC_TCP_STAGING_DEPTH` (planned) | `4` | Staging chunks in flight per worker |
| `MC_TCP_ZC` (planned) | `0` | Enable dmabuf zero copy between GPU memory and the NIC (io_uring zero-copy receive, devmem TCP transmit); needs kernel 6.16+ and a header/data-split capable NIC |
| `MC_TCP_ZC_IFACE`, `MC_TCP_ZC_RXQS` (planned) | unset | Interface and RX queue list steered to the zero-copy path |
| `MC_TCP_ZCRX_AREA_MB` (planned) | `512` | GPU memory per zero-copy RX queue |
| `MC_DPDK_PORTS`, `MC_DPDK_LCORES`, `MC_DPDK_EAL_ARGS`, `MC_DPDK_IP` (planned) | unset | DPDK kernel-bypass transport (`mooncake_protocol: "dpdk"`); setting `MC_DPDK_PORTS` enables it |

### io_uring inside containers

Docker 25 and later, and runtimes that copy its default seccomp profile, block the `io_uring_setup`, `io_uring_enter` and `io_uring_register` syscalls. Mooncake then falls back to the asio data plane with a warning, and `VLLM_KV_OFFLOAD_DISK_IO_BACKEND=io_uring` falls back to POSIX I/O. To use io_uring, run the container with a seccomp profile that allows those three syscalls (`docker run --security-opt seccomp=<profile.json> ...`), or on Kubernetes set `securityContext.seccompProfile` for the pod accordingly.

### Disk KV offload tiers

`VLLM_KV_OFFLOAD_DISK_IO_BACKEND` (default `posix`) selects how the disk KV offload tiers issue block I/O: the disk mode of the simple KV offload worker (`VLLM_USE_SIMPLE_KV_OFFLOAD`) and the `fs` secondary tier of [KV offloading](kv_offloading_usage.md). With `io_uring`, block reads and writes go through Mooncake's `mooncake.uring` module: the pinned staging slots (or the host KV buffer) are registered once as io_uring fixed buffers, the file keeps `O_DIRECT`, and each staged block is submitted as one batch instead of one `preadv`/`pwritev` call per slot. When `mooncake.uring` is missing, reports `SUPPORT_URING = False`, or io_uring cannot be set up, vLLM logs a warning and uses the POSIX path.

## Example Scripts/Code

Refer to these example scripts in the vLLM repository:

- [run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh)
- [mooncake_connector_proxy.py](../../examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py)
