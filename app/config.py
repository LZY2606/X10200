"""集中配置：固定容差，保证数值结果可复现。"""

import os


# --- 固定数值容差（所有算法共用，禁止按数据自适应）---
SINGULAR_DET_TOL = 1.0e-12
INDEX_TOL = 0.25
FAMILY_TRANSFORM_TOL = 1.0e-6
UNIMODULAR_TOL = 1.0e-6
REDUCE_MAX_STEPS = 10_000

# 约化基数值去重容差（严格；近但超差的候选不合并）
FAMILY_VECTOR_TOL = 1.0e-9

# 候选枚举
MIN_INDEXED = 6
MAX_CANDIDATES = 120

DB_PATH = os.environ.get("LCC_DB_PATH", "lattice_chamber.db")
