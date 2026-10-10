import os
from contextlib import contextmanager
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from labtrack.validation import FIELDS, validate, validate_metrics

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'data' / 'labtrack.db'


class ExperimentStore:
    """Transactional SQLite repository, including compatible startup migrations."""
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get('LABTRACK_DB', DEFAULT_DB))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, model TEXT NOT NULL, dataset TEXT NOT NULL,
                batch_size INTEGER NOT NULL CHECK(batch_size > 0),
                learning_rate REAL NOT NULL CHECK(learning_rate > 0),
                epochs INTEGER NOT NULL CHECK(epochs > 0), gpu TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL, train_command TEXT NOT NULL,
                eval_command TEXT NOT NULL,
                success_rate REAL CHECK(success_rate BETWEEN 0 AND 100), notes TEXT NOT NULL,
                created_time TEXT NOT NULL, updated_time TEXT NOT NULL)''')
            # Serialize startup migrations; never recreate the experiments table.
            conn.execute('BEGIN IMMEDIATE')
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(experiments)')}
            if 'status' not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN status TEXT NOT NULL DEFAULT 'planned' CHECK(status IN ('planned','running','completed','failed'))")
            conn.execute('''CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS experiment_tags (
                experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY(experiment_id,tag_id))''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tag_experiment ON experiment_tags(tag_id,experiment_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON experiments(created_time)')
            conn.execute('''CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                step INTEGER NOT NULL CHECK(step >= 0),
                metric_name TEXT NOT NULL CHECK(metric_name IN ('loss','learning_rate','accuracy','success_rate')),
                metric_value REAL NOT NULL CHECK(metric_value >= 0),
                UNIQUE(experiment_id, step, metric_name))''')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, values):
        return self.create_many([values])[0]

    def create_many(self, values):
        """Validate every experiment before one transaction inserts experiments and tags."""
        cleaned = [validate(value) for value in values]
        ids = []
        with self.connect() as conn:
            for data in cleaned:
                ids.append(self._insert_experiment(conn, data))
        return ids

    @classmethod
    def _insert_experiment(cls, conn, data):
        """Insert validated fields and tag relations using the caller's transaction."""
        now = datetime.now(timezone.utc).isoformat(timespec='microseconds')
        cursor = conn.execute(
            f"INSERT INTO experiments ({','.join(FIELDS)},created_time,updated_time) VALUES ({','.join(['?'] * (len(FIELDS)+2))})",
            [data[k] for k in FIELDS] + [now, now])
        cls._set_tags(conn, cursor.lastrowid, data['tags'])
        return cursor.lastrowid

    def clone(self, experiment_id):
        """Copy configuration and tags, resetting state/results without copying metrics."""
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._require_experiment(conn, experiment_id)
            source = self._with_tags(conn, [conn.execute(
                'SELECT * FROM experiments WHERE id=?', (experiment_id,)).fetchone()])[0]
            base = source['name'] + '_copy'
            name, suffix = base, 2
            while conn.execute('SELECT 1 FROM experiments WHERE name=?', (name,)).fetchone():
                name = f'{base}_{suffix}'
                suffix += 1
            data = validate(dict(source, name=name, status='planned', success_rate=None))
            return self._insert_experiment(conn, data)

    @staticmethod
    def _set_tags(conn, experiment_id, tags):
        conn.execute('DELETE FROM experiment_tags WHERE experiment_id=?', (experiment_id,))
        for name in tags:
            conn.execute('INSERT INTO tags(name) VALUES (?) ON CONFLICT(name) DO NOTHING', (name,))
            tag_id = conn.execute('SELECT id FROM tags WHERE name=?', (name,)).fetchone()['id']
            conn.execute('INSERT INTO experiment_tags(experiment_id,tag_id) VALUES (?,?)', (experiment_id, tag_id))

    @staticmethod
    def _with_tags(conn, rows):
        experiments = [dict(row) for row in rows]
        names = {}
        for row in conn.execute('SELECT et.experiment_id,t.name FROM experiment_tags et JOIN tags t ON t.id=et.tag_id ORDER BY t.name'):
            names.setdefault(row['experiment_id'], []).append(row['name'])
        for row in experiments:
            row['tags'] = names.get(row['id'], [])
        return experiments

    def get(self, experiment_id):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM experiments WHERE id=?', (experiment_id,)).fetchone()
            return self._with_tags(conn, [row])[0] if row else None

    def update(self, experiment_id, values):
        current = self.get(experiment_id)
        if current is None:
            raise ValueError('实验不存在，可能已经被删除')
        data = validate(dict(current, **values))
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE experiments SET {','.join(k+'=?' for k in FIELDS)},updated_time=? WHERE id=?",
                [data[k] for k in FIELDS] + [datetime.now(timezone.utc).isoformat(timespec='microseconds'), experiment_id])
            if cursor.rowcount == 0:
                raise ValueError('实验不存在，可能已经被删除')
            self._set_tags(conn, experiment_id, data['tags'])

    def delete(self, experiment_id):
        with self.connect() as conn:
            cursor = conn.execute('DELETE FROM experiments WHERE id=?', (experiment_id,))
            if cursor.rowcount == 0:
                raise ValueError('实验不存在，可能已经被删除')

    def list(self, name='', model='', start=None, end=None, status='', tag=''):
        clauses, params = [], []
        for key, value in [('name', name), ('model', model)]:
            if value:
                clauses.append(f"{key} LIKE ? ESCAPE '\\'")
                escaped = value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                params.append('%' + escaped + '%')
        if start:
            clauses.append('substr(created_time,1,10)>=?')
            params.append(str(start))
        if end:
            clauses.append('substr(created_time,1,10)<=?')
            params.append(str(end))
        if status:
            clauses.append('status=?')
            params.append(status)
        if tag:
            clauses.append('id IN (SELECT et.experiment_id FROM experiment_tags et JOIN tags t ON t.id=et.tag_id WHERE t.name=?)')
            params.append(tag)
        query = 'SELECT * FROM experiments'
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)
        with self.connect() as conn:
            return self._with_tags(conn, conn.execute(query + ' ORDER BY created_time DESC,id DESC', params))

    def add_metrics(self, experiment_id, metrics, replace=False):
        """Atomic batch upsert; optionally replace all existing experiment metrics."""
        cleaned = validate_metrics(metrics)
        if not cleaned:
            raise ValueError('没有可保存的指标，现有数据未修改')
        with self.connect() as conn:
            self._require_experiment(conn, experiment_id)
            if replace:
                conn.execute('DELETE FROM experiment_metrics WHERE experiment_id=?', (experiment_id,))
            conn.executemany(
                """INSERT INTO experiment_metrics (experiment_id,step,metric_name,metric_value)
                   VALUES (?,?,?,?) ON CONFLICT(experiment_id,step,metric_name)
                   DO UPDATE SET metric_value=excluded.metric_value""",
                [(experiment_id, step, name, value) for step, name, value in cleaned])
            self._touch(conn, experiment_id)

    def get_metrics(self, experiment_id):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                'SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY step,metric_name',
                (experiment_id,))]

    def delete_metrics(self, experiment_id):
        with self.connect() as conn:
            self._require_experiment(conn, experiment_id)
            conn.execute('DELETE FROM experiment_metrics WHERE experiment_id=?', (experiment_id,))
            self._touch(conn, experiment_id)

    @staticmethod
    def _require_experiment(conn, experiment_id):
        if conn.execute('SELECT id FROM experiments WHERE id=?', (experiment_id,)).fetchone() is None:
            raise ValueError('实验不存在，可能已经被删除')

    @staticmethod
    def _touch(conn, experiment_id):
        conn.execute('UPDATE experiments SET updated_time=? WHERE id=?',
                     (datetime.now(timezone.utc).isoformat(timespec='microseconds'), experiment_id))

    def list_tags(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT t.id,t.name,COUNT(et.experiment_id) AS experiment_count FROM tags t
                   LEFT JOIN experiment_tags et ON t.id=et.tag_id GROUP BY t.id
                   ORDER BY experiment_count DESC,t.name""")]

    def get_all_metrics(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                'SELECT experiment_id,step,metric_name,metric_value FROM experiment_metrics ORDER BY experiment_id,step,metric_name')]

    def cleanup_unused_tags(self):
        with self.connect() as conn:
            return conn.execute('DELETE FROM tags WHERE NOT EXISTS (SELECT 1 FROM experiment_tags WHERE tag_id=tags.id)').rowcount
