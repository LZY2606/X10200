# 晶格候选室 (Lattice Candidate Room)

从衍射反射向量审阅多个晶格候选，而不是只接受软件吐出的单一晶胞。

## 安装与运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5260
# 打开 http://127.0.0.1:5260 ，点击“载入演示数据”
```

计算全程不访问外网；数据库路径可用环境变量 `LATTICE_DB` 覆盖（默认 `lattice.db`）。

## 结构

- `lattice/core.py` — 晶胞约化（记录完整幺模变换链）、度量张量比较，固定容差
- `lattice/indexing.py` — 候选生成（最短 |q| 种子三元组 + 多晶域二次播种）、
  hkl 索引、协方差加权精修、离群分类、四分量评分（已索引比例 / 加权残差 /
  复杂度 / 离群预算，永不合并为单一总分）
- `lattice/families.py` — 等价晶胞经幺模基变换归族；变换矩阵与原候选都保留，
  超出固定容差的近似候选不合并
- `db.py` — SQLite：原始观测、分析、候选、反射对应、人工标记、冻结；
  标记只使依赖候选过期，校准变更使全部分候选过期；冻结用版本号做乐观并发
- `app.py` — FastAPI 接口；`static/index.html` — 倒易空间投影、候选族、
  反射对应、残差分布，点击 hkl 回溯观测与变换链

## 主要接口

- `POST /api/observations/import` — 导入峰坐标、强度、协方差、校准版本（存原始观测）
- `POST /api/analyses` / `GET /api/analyses/{id}` — 创建并运行分析 / 查看完整状态
- `POST /api/analyses/{id}/marks` — 锁定或排除反射（局部过期）
- `POST /api/analyses/{id}/calibration` — 调整校准点（全部过期）
- `POST /api/analyses/{id}/fork` — 从当前分析分叉
- `POST /api/analyses/{id}/freeze` — 冻结解释（钉住校准/容差/人工标记；旧草案返回 409）
