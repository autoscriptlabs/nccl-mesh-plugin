# sparkrun integration

This example stages the plugin in a path that sparkrun-mounted containers can
see and adds the tested NCCL environment to an existing recipe.

It is intentionally an integration fragment rather than a model-specific
recipe.

## Stage the binary

On every host:

```bash
mkdir -p ~/.cache/huggingface/nccl-mesh
cp /path/to/nccl-mesh-plugin/libnccl-net.so \
  ~/.cache/huggingface/nccl-mesh/libnccl-net.so
```

Verify one identical hash:

```bash
for host in spark-a spark-b spark-c spark-d; do
  ssh "$host" 'sha256sum ~/.cache/huggingface/nccl-mesh/libnccl-net.so'
done
```

With the standard Hugging Face cache mount used in the validated deployment,
the container path is:

```text
/cache/huggingface/nccl-mesh/libnccl-net.so
```

## Add the environment block

Merge [mesh-env.yaml](mesh-env.yaml) under the recipe's existing `env:` mapping.
Replace the interface name, plugin path, and NCCL library path for the local
image.

The tested profile uses:

- exact external NCCL library preloaded from the image;
- Mesh plugin selected by full container path;
- shared management interface for bootstrap and hybrid TCP;
- physical-ring collective preference;
- 900-second plugin operation timeout for uneven model startup;
- periodic plugin latency metrics disabled.

## Verify inside a worker

```bash
env | grep -E '^NCCL_(NET|NET_PLUGIN|SOCKET_IFNAME|ALGO)|^NCCL_MESH_'
ls -l /cache/huggingface/nccl-mesh/libnccl-net.so
sha256sum /cache/huggingface/nccl-mesh/libnccl-net.so
```

Expected NCCL startup:

```text
NET/Plugin: Loaded net plugin Mesh (v9)
Using network Mesh
```

## Benchmark an already-running service

When using `--skip-run`, the API model name may differ from the recipe's
container-local model path. Pass the served API model name and a host-accessible
tokenizer explicitly to llama-benchy through sparkrun.

Example shape:

```bash
sparkrun benchmark performance recipe.yaml \
  --skip-run \
  --no-stop \
  --hosts host-a,host-b,host-c,host-d \
  --tp 4 \
  --port 8000 \
  --output /tmp/mesh-benchmark.yaml \
  -b model='served/api-model-name' \
  -b served_model_name='served/api-model-name' \
  -b tokenizer='/host/path/to/tokenizer' \
  -b pp=2048 \
  -b tg=128 \
  -b depth=0,4096,16384 \
  -b concurrency=1,2,4 \
  -b runs=3
```

Without the served-model override, the benchmark client can send the
container-local path as the OpenAI API model ID and receive HTTP 404.
