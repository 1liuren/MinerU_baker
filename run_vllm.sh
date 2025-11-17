#!/bin/bash
set -e  # 遇到错误立即退出

# 禁用代理，避免内网请求通过代理导致连接问题
export no_proxy="localhost,127.0.0.1,10.10.50.53,10.10.50.55,10.10.50.0/24"
export NO_PROXY="localhost,127.0.0.1,10.10.50.53,10.10.50.55,10.10.50.0/24"
# 如果需要完全禁用代理，可以取消下面两行的注释
# unset http_proxy
# unset https_proxy

sudo -v
# 每隔 5 分钟自动刷新一次 sudo 时效，直到脚本结束
while true; do sudo -v; sleep 300; done &

# 55_0818
sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250825 \
  -o data_55/2025自有库_55_0825_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/suanfa/20250825 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

# 55 qckj 20250819
sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250823 \
  -o data_55/2025自有库_55_0823_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250823 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250825 \
  -o data_55/2025自有库_55_0825_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250825 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250829 \
  -o data_55/2025自有库_55_0829_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250829 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250901 \
  -o data_55/2025自有库_55_0901_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250901 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250902 \
  -o data_55/2025自有库_55_0902_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250902 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250908 \
  -o data_55/2025自有库_55_0908_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250908 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250910 \
  -o data_55/2025自有库_55_0910_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250910 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250913 \
  -o data_55/2025自有库_55_0913_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250913 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300

sudo -E $(which python) scripts/main.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250915 \
  -o data_55/2025自有库_55_0915_qckj_vllm \
  --batch-size 1 --concurrent-batches 30 \
  -d /data/liufan/coder-server-55/qckj_suanfa/20250915 \
  --server-url http://10.10.50.53:30000,http://10.10.50.55:30000 \
  --batches-per-round 300
