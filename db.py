import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  parent_id INTEGER,
  revision INTEGER NOT NULL DEFAULT 0,
  calibration_version TEXT NOT NULL,
  cal_scale REAL NOT NULL DEFAULT 1.0,
  cal_ox REAL NOT NULL DEFAULT 0,
  cal_oy REAL NOT NULL DEFAULT 0,
  cal_oz REAL NOT NULL DEFAULT 0,
  tolerance REAL NOT NULL DEFAULT 0.18,
  outlier_budget INTEGER NOT NULL DEFAULT 5,
  created TEXT
);
CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_id INTEGER NOT NULL,
  qx REAL NOT NULL, qy REAL NOT NULL, qz REAL NOT NULL,
  intensity REAL NOT NULL DEFAULT 1,
  cov TEXT NOT NULL,
  locked INTEGER NOT NULL DEFAULT 0,
  excluded INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS families(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_id INTEGER NOT NULL,
  metric TEXT NOT NULL,
  rep_basis TEXT,
  representative INTEGER
);
CREATE TABLE IF NOT EXISTS candidates(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_id INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  basis TEXT NOT NULL,
  reduced_basis TEXT NOT NULL,
  metric TEXT NOT NULL,
  family_id INTEGER NOT NULL,
  family_transform TEXT,
  revision INTEGER NOT NULL,
  calib_version TEXT NOT NULL,
  deps TEXT NOT NULL,
  stale INTEGER NOT NULL DEFAULT 0,
  scores TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id INTEGER NOT NULL,
  observation_id INTEGER NOT NULL,
  h INTEGER NOT NULL, k INTEGER NOT NULL, l INTEGER NOT NULL,
  px REAL NOT NULL, py REAL NOT NULL, pz REAL NOT NULL,
  residual REAL NOT NULL,
  weighted REAL NOT NULL,
  outlier INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS transforms(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id INTEGER NOT NULL,
  step INTEGER NOT NULL,
  matrix TEXT NOT NULL,
  description TEXT
);
CREATE TABLE IF NOT EXISTS freezes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_id INTEGER NOT NULL,
  revision INTEGER NOT NULL,
  snapshot TEXT NOT NULL,
  created TEXT
);
"""


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
