"""Run from project root: python -m scripts.seed."""
import json
from pathlib import Path
from labtrack.database import ExperimentStore


def main():
    store = ExperimentStore()
    if store.list():
        print('数据库已有实验，跳过导入以防重复。')
        return
    rows = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'examples.json').read_text(encoding='utf-8'))
    for row in rows:
        store.create(row)
    print(f'已导入 {len(rows)} 个示例实验。')


if __name__ == '__main__':
    main()
