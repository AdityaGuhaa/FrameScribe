"""SQLite database manager for VideoAI projects."""

import aiosqlite
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: str = 'data/videoai.db'):
        self.db_path = db_path

    async def init(self) -> None:
        """Initialize the database directory and projects table if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    video_file TEXT,
                    created_at TEXT,
                    duration REAL,
                    status TEXT,
                    result_path TEXT
                )
            ''')
            await db.commit()
        logger.info("Database initialized")

    async def save_project(self, job_id: str, video_file: str, duration: float, status: str, result_path: str) -> None:
        """Insert or update a project record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO projects (id, video_file, created_at, duration, status, result_path)
                VALUES (?, ?, datetime('now'), ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    result_path=excluded.result_path,
                    duration=excluded.duration
            ''', (job_id, video_file, duration, status, result_path))
            await db.commit()

    async def get_projects(self) -> List[Dict]:
        """Return all projects ordered by created_at descending."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('SELECT * FROM projects ORDER BY created_at DESC')
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_project(self, project_id: str) -> Optional[Dict]:
        """Get a single project by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_project(self, project_id: str) -> None:
        """Delete a project record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('DELETE FROM projects WHERE id = ?', (project_id,))
            await db.commit()

    async def update_status(self, job_id: str, status: str) -> None:
        """Update the status of a project."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE projects SET status = ? WHERE id = ?', (status, job_id))
            await db.commit()
