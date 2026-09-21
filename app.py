"""uvicorn 入口：``.venv/bin/uvicorn app:app``。"""

from app.web import create_app

app = create_app()
