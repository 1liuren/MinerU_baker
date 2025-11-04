#!/bin/bash
# 数据整理脚本使用示例
# 根据您的实际环境调整以下路径参数

# ========================================
# 非55服务器数据
# ========================================

# 2025自有库_0807_sglang
# python scripts/reorganize_data.py \
#   -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250807 \
#   -o data_55/2025自有库_0807_sglang \
#   --organized-output /data/Ldata/pdf_liufan/ \
#   --data-json /data/liufan/coder-server/suanfa/20250807 \
#   --levels 4 \
#   --log-level INFO \
#   --max-workers 32

# 2025自有库_0807_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250807 \
  -o data_55/2025自有库_0807_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/suanfa/20250807 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0815_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250815 \
  -o data_55/2025自有库_0815_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/suanfa/20250815 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0818_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250818 \
  -o data_55/2025自有库_0818_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/suanfa/20250818 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0819_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250819 \
  -o data_55/2025自有库_0819_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/suanfa/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0819_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250819 \
  -o data_55/2025自有库_0819_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/suanfa_back/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0819_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/qckj/origin/20250819 \
  -o data_55/2025自有库_0819_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/qckj_suanfa/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_0819_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server/qckj/origin/20250819 \
  -o data_55/2025自有库_0819_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server/qckj_suanfa_back/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# ========================================
# 55服务器数据
# ========================================

# 2025自有库_55_0815_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250815 \
  -o data_55/2025自有库_55_0815_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/suanfa/20250815 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0818_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250818 \
  -o data_55/2025自有库_55_0818_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/suanfa/20250818 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0819_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250819 \
  -o data_55/2025自有库_55_0819_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/suanfa/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0825_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250825 \
  -o data_55/2025自有库_55_0825_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/suanfa/20250825 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# ========================================
# 55服务器 - qckj数据
# ========================================

# 2025自有库_55_0819_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250819 \
  -o data_55/2025自有库_55_0819_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0819_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250819 \
  -o data_55/2025自有库_55_0819_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250819 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0823_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250823 \
  -o data_55/2025自有库_55_0823_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250823 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0823_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250823 \
  -o data_55/2025自有库_55_0823_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250823 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0825_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250825 \
  -o data_55/2025自有库_55_0825_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250825 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0825_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250825 \
  -o data_55/2025自有库_55_0825_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250825 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0829_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250829 \
  -o data_55/2025自有库_55_0829_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250829 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0829_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250829 \
  -o data_55/2025自有库_55_0829_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250829 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0901_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250901 \
  -o data_55/2025自有库_55_0901_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250901 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0901_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250901 \
  -o data_55/2025自有库_55_0901_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250901 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0902_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250902 \
  -o data_55/2025自有库_55_0902_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250902 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0902_qckj_vllm_back (回捞数据)
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250902 \
  -o data_55/2025自有库_55_0902_qckj_vllm_back \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa_back/20250902 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0908_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250908 \
  -o data_55/2025自有库_55_0908_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250908 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0910_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250910 \
  -o data_55/2025自有库_55_0910_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250910 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0911_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250911 \
  -o data_55/2025自有库_55_0911_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250911 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0913_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250913 \
  -o data_55/2025自有库_55_0913_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250913 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0915_qckj_sglang
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250915 \
  -o data_55/2025自有库_55_0915_qckj_sglang \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250915 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0916_qckj_sglang
# python scripts/reorganize_data.py \
#   -i /data/liufan/coder-server-55/qckj/origin/20250916 \
#   -o data_55/2025自有库_55_0916_qckj_sglang \
#   --organized-output /data/Ldata/pdf_liufan/ \
#   --data-json /data/liufan/coder-server-55/qckj_suanfa/20250916 \
#   --levels 4 \
#   --log-level INFO \
#   --max-workers 32

# 2025自有库_55_0916_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250916 \
  -o data_55/2025自有库_55_0916_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250916 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0917_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250917 \
  -o data_55/2025自有库_55_0917_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250917 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0918_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250918 \
  -o data_55/2025自有库_55_0918_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250918 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0922_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250922 \
  -o data_55/2025自有库_55_0922_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250922 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0923_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250923 \
  -o data_55/2025自有库_55_0923_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250923 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0927_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250927 \
  -o data_55/2025自有库_55_0927_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250927 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32

# 2025自有库_55_0928_qckj_vllm
python scripts/reorganize_data.py \
  -i /data/liufan/coder-server-55/qckj/origin/20250928 \
  -o data_55/2025自有库_55_0928_qckj_vllm \
  --organized-output /data/Ldata/pdf_liufan/ \
  --data-json /data/liufan/coder-server-55/qckj_suanfa/20250928 \
  --levels 4 \
  --log-level INFO \
  --max-workers 32
