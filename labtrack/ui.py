"""Streamlit presentation and interactions; parsing and storage live in dedicated modules."""
import hashlib
import re
from collections import Counter
import pandas as pd
import plotly.express as px
import streamlit as st
from labtrack.validation import BASE_FIELDS, FIELDS, STATUSES

LABELS = dict(zip(BASE_FIELDS, ['实验名称', '模型', '数据集', 'Batch size', 'Learning rate', 'Epochs', 'GPU', 'Checkpoint path', 'Train command', 'Eval command', 'Success rate (%)', '备注']))
LABELS['status'] = 'Status'


def experiment_form(key, existing=None):
    data = existing or {}
    with st.form(key, clear_on_submit=existing is None):
        left, right = st.columns(2)
        values = {}
        for column, keys in [(left, ['name', 'model', 'dataset', 'batch_size', 'learning_rate', 'epochs']),
                             (right, ['gpu', 'checkpoint_path', 'train_command', 'eval_command', 'notes'])]:
            with column:
                for field in keys:
                    if field in ['batch_size', 'epochs']:
                        values[field] = st.number_input(LABELS[field], min_value=1, value=int(data.get(field, 32 if field == 'batch_size' else 10)), step=1)
                    elif field == 'learning_rate':
                        values[field] = st.number_input(LABELS[field], min_value=1e-12, value=float(data.get(field, 0.0001)), format='%.12f')
                    elif field in ['train_command', 'eval_command', 'notes']:
                        values[field] = st.text_area(LABELS[field], value=data.get(field, ''))
                    else:
                        values[field] = st.text_input(LABELS[field], value=data.get(field, ''))
        values['status'] = st.selectbox('Status', STATUSES, index=STATUSES.index(data.get('status', 'planned')), key=f'status_{key}')
        tags = st.text_area('Tags（逗号或换行分隔）', value='\n'.join(data.get('tags', [])), key=f'tags_{key}')
        values['tags'] = re.split(r'[,，\n]', tags)
        measured = st.checkbox('已完成评测（取消勾选表示尚无结果）', value=data.get('success_rate') is not None)
        rate = st.number_input('Success rate (%)', min_value=0.0, max_value=100.0, value=float(data.get('success_rate') or 0), step=0.1)
        values['success_rate'] = rate if measured else None
        st.caption('名称、模型、数据集必填。命令和路径仅保存为文本。')
        submitted = st.form_submit_button('保存实验', type='primary')
    return values if submitted else None


def dashboard(rows):
    st.header('Dashboard')
    df = pd.DataFrame(rows)
    a, b, c = st.columns(3)
    a.metric('实验总数', len(rows))
    rates = [r['success_rate'] for r in rows if r['success_rate'] is not None]
    b.metric('最佳 Success rate', f'{max(rates):.2f}%' if rates else '暂无结果')
    c.metric('已评测实验', len(rates))
    counts = {status: sum(row.get('status', 'planned') == status for row in rows) for status in STATUSES}
    for column, status in zip(st.columns(4), STATUSES):
        column.metric(status.title(), counts[status])
    if not rows:
        st.info('还没有实验，请从侧栏新建实验，或按 README 导入示例数据。')
        return
    distribution = pd.DataFrame({'Status': list(counts), '实验数': list(counts.values())})
    st.plotly_chart(px.bar(distribution, x='Status', y='实验数', title='实验状态分布'), use_container_width=True)
    tag_counts = Counter(tag for row in rows for tag in row.get('tags', []))
    if tag_counts:
        st.subheader('最常用标签')
        st.dataframe(pd.DataFrame(tag_counts.most_common(10), columns=['标签', '实验数']), hide_index=True, use_container_width=True)
    st.subheader('最近实验')
    st.dataframe(df[['id', 'name', 'model', 'dataset', 'status', 'success_rate', 'created_time']].head(5), hide_index=True, use_container_width=True)
    measured = df.dropna(subset=['success_rate']).copy()
    if not measured.empty:
        measured['实验'] = measured['name'] + ' (#' + measured['id'].astype(str) + ')'
        st.plotly_chart(px.bar(measured, x='实验', y='success_rate', color='model', labels={'success_rate': 'Success rate (%)'}), use_container_width=True)
    daily = df.assign(日期=df.created_time.str[:10]).groupby('日期').size().reset_index(name='实验数')
    st.plotly_chart(px.line(daily, x='日期', y='实验数', markers=True), use_container_width=True)
    st.caption('不同数据集和评测协议下的 success rate 不一定可直接比较。所有日期按 UTC 保存和筛选。')


def details(row):
    """Render overview fields without command or metric sections."""
    st.table(pd.DataFrame({'字段': [LABELS[k] for k in FIELDS[:8]], '值': [str(row[k]) for k in FIELDS[:8]]}))
    st.write(f"Status：{row.get('status', 'planned')}")
    st.text('Tags：' + (' · '.join(row.get('tags', [])) or '无标签'))
    st.metric('Success rate', '尚未评测' if row['success_rate'] is None else f"{row['success_rate']:.2f}%")
    st.text(row['notes'] or '暂无备注')
    st.caption(f"创建时间 (UTC)：{row['created_time']} | 更新时间 (UTC)：{row['updated_time']}")


def logs_and_metrics(store, row):
    """Keep upload, preview and plots local to the experiment detail UI."""
    experiment_id = row['id']
    st.subheader('日志与指标')
    _log_upload(store, experiment_id)
    metrics = store.get_metrics(experiment_id)
    _metric_charts(store, experiment_id, metrics)


def _log_upload(store, experiment_id):
    """Preview a text log and save only after an explicit button click."""
    from labtrack.parser import parse_uploaded_log
    st.caption('支持 UTF-8 的 .txt / .log（最大 5 MB），仅解析文本。Accuracy / Success Rate：百分号值保持原值；无百分号的 0–1 值按比例转换为百分比，其他值按百分比处理。')
    uploaded = st.file_uploader('上传训练日志', type=['txt', 'log'], key=f'log_{experiment_id}')
    if uploaded is not None:
        try:
            result = parse_uploaded_log(uploaded.getvalue(), uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.caption(f'识别 {result.parsed_lines} 行、{len(result.metrics)} 条指标；跳过 {result.skipped_lines} 行。')
            if result.warnings:
                with st.expander(f'解析提示（{len(result.warnings)} 条）'):
                    for warning in result.warnings[:50]:
                        st.write(warning)
                    if len(result.warnings) > 50:
                        st.caption('仅展示前 50 条提示。')
            if not result.metrics:
                st.warning('没有解析出指标。请确认每行包含 Epoch 和 Loss / LR / Accuracy / Success Rate，数值使用冒号或等号分隔。')
            else:
                st.dataframe(pd.DataFrame(result.metrics), hide_index=True, use_container_width=True)
                replace = st.checkbox('保存时替换该实验的全部已有指标', key=f'replace_{experiment_id}')
                st.caption('默认合并：相同 Epoch 和指标名称更新原值，其他记录保留。替换模式会清除旧指标。')
                if st.button('保存日志指标', key=f'import_{experiment_id}'):
                    store.add_metrics(experiment_id, result.metrics, replace=replace)
                    st.session_state.notice = f'已保存 {len(result.metrics)} 条指标'
                    st.rerun()


def _metric_charts(store, experiment_id, metrics):
    """Render only persisted metric series, summaries and confirmed clearing."""
    from labtrack.report import summarize_metrics, SUMMARY_LABELS
    if not metrics:
        st.info('暂无已保存的训练指标。上传日志并保存后即可查看曲线。')
    else:
        summary = summarize_metrics(metrics)
        columns = st.columns(len(summary)) if summary else []
        for column, (key, value) in zip(columns, summary.items()):
            column.metric(SUMMARY_LABELS[key], f'{value:g}')
        st.caption('Final 指该指标最大 Epoch 的值；Best Loss 指最小 Loss。')
        df = pd.DataFrame(metrics)
        titles = {'loss': 'Loss', 'learning_rate': 'Learning Rate',
                  'accuracy': 'Accuracy (%)', 'success_rate': 'Success Rate (%)'}
        for name, title in titles.items():
            points = df[df.metric_name == name]
            if not points.empty:
                st.plotly_chart(px.line(points, x='step', y='metric_value', markers=True,
                                        title=f'{title} vs Epoch', labels={'step': 'Epoch', 'metric_value': title}),
                                use_container_width=True, key=f'chart_{experiment_id}_{name}')
        with st.expander('已保存指标数据'):
            st.dataframe(df[['step', 'metric_name', 'metric_value']], hide_index=True, use_container_width=True)
        with st.expander('清空训练指标'):
            confirm = st.checkbox('确认清空该实验所有训练指标', key=f'clear_confirm_{experiment_id}')
            if st.button('清空指标', disabled=not confirm, key=f'clear_{experiment_id}'):
                store.delete_metrics(experiment_id)
                st.session_state.notice = '训练指标已清空'
                st.rerun()


def command_parameters(row):
    from labtrack.command_parser import parse_train_command, check_consistency
    st.subheader('训练命令参数')
    try:
        arguments = parse_train_command(row['train_command'])
    except ValueError as exc:
        st.warning(str(exc))
        return
    if arguments:
        st.table(pd.DataFrame({'参数': list(arguments), '值': [str(value) for value in arguments.values()]}))
    else:
        st.info('没有识别到长参数（--key）。')
    checks = check_consistency(row, arguments)
    for check in checks:
        if not check['matches']:
            st.warning(f"{check['label']} 不一致：记录值：{check['recorded']}；训练命令 --{check['parameter']}：{check['command']}" +
                       ('（命令参数值无效）' if not check['valid'] else ''))
    if checks and all(check['matches'] for check in checks):
        st.success('训练命令中已提供的待核对参数均与实验记录一致。')
    elif not checks:
        st.caption('命令未提供 batch_size、learning_rate / lr、epochs 或 gpu，无法确认一致性。')
    st.caption('仅解析文本和提示差异，不执行命令，不自动修改实验记录。')


def csv_management(store):
    from labtrack.import_export import export_experiments, export_metrics, parse_experiments_csv
    st.header('CSV 导入与导出')
    st.subheader('导出全部数据')
    st.download_button('下载 experiments.csv', export_experiments(store.list()),
                       file_name='experiments.csv', mime='text/csv', key='export_experiments')
    st.download_button('下载 experiment_metrics.csv', export_metrics(store.get_all_metrics()),
                       file_name='experiment_metrics.csv', mime='text/csv', key='export_metrics')
    st.subheader('导入实验基础信息')
    st.caption('仅新增实验，不导入指标；CSV 中的 ID 和时间忽略，重新生成。文件必须为 UTF-8，最大 5 MB；Tags 用 | 分隔。')
    uploaded = st.file_uploader('上传 experiments.csv', type=['csv'], key='csv_upload')
    if uploaded is None:
        return
    content = uploaded.getvalue()
    result = parse_experiments_csv(content)
    if result.preview:
        st.dataframe(pd.DataFrame(result.preview), hide_index=True, use_container_width=True)
    if result.errors:
        st.error(f'验证失败，共 {len(result.errors)} 个错误；整个文件不会写入任何数据。')
        for message in result.errors[:100]:
            st.text(message)
        if len(result.errors) > 100:
            st.caption('页面仅显示前 100 个错误。')
        return
    st.success(f'全部 {len(result.rows)} 行验证通过。')
    digest = hashlib.sha256(content).hexdigest()
    confirm = st.checkbox('确认新增以上实验（不会覆盖已有记录）', key=f'csv_confirm_{digest}')
    if st.button('确认导入', disabled=not confirm, key='csv_import'):
        ids = store.create_many(result.rows)
        st.session_state.notice = f'已一次性导入 {len(ids)} 个实验'
        st.rerun()

def experiment_detail(store, row):
    """Group existing detail interactions into tabs without changing persistence rules."""
    st.subheader(f"{row['name']} · #{row['id']}")
    overview, metrics, commands, data = st.tabs(['Overview', 'Metrics', 'Commands', 'Data'])
    with overview:
        details(row)
        with st.expander('编辑实验'):
            values = experiment_form(f"edit_{row['id']}", row)
            if values is not None:
                store.update(row['id'], values)
                st.session_state.notice = '实验已更新'
                st.rerun()
    with metrics:
        logs_and_metrics(store, row)
    with commands:
        for key in ['train_command', 'eval_command']:
            st.write(LABELS[key])
            st.code(row[key] or '未填写', language='bash')
        command_parameters(row)
    with data:
        _experiment_actions(store, row)


def _experiment_actions(store, row):
    """Download the report, clone configuration, or confirm permanent deletion."""
    from labtrack.report import generate_report
    experiment_id = row['id']
    st.download_button('导出实验报告', data=generate_report(row, store.get_metrics(experiment_id)),
                       file_name=f'labtrack_experiment_{experiment_id}.md', mime='text/markdown',
                       key=f'report_{experiment_id}')
    st.caption('克隆复制配置与标签，新实验为 planned，评测结果为空，不复制训练指标。')
    if st.button('克隆实验', key=f'clone_{experiment_id}'):
        cloned_id = store.clone(experiment_id)
        st.session_state.open_experiment = cloned_id
        st.session_state.notice = f'已克隆为实验 #{cloned_id}，可在 Overview 中继续编辑'
        st.rerun()
    with st.expander('删除实验'):
        confirm = st.checkbox(f"确认永久删除 #{experiment_id} · {row['name']}", key=f'confirm_{experiment_id}')
        if st.button('删除实验', disabled=not confirm, key=f'delete_{experiment_id}'):
            store.delete(experiment_id)
            st.session_state.notice = '实验已删除'
            st.rerun()


def experiment_comparison(store):
    """Render the existing two-experiment parameter and result comparison."""
    st.header('比较两个实验')
    rows = store.list()
    if len(rows) < 2:
        st.info('至少需要两个实验才能比较')
        return
    mapping = {r['id']: r for r in rows}
    selected = st.multiselect('选择两个实验', list(mapping), max_selections=2, format_func=lambda i: f"#{i} · {mapping[i]['name']}")
    if len(selected) != 2:
        st.info('请选择恰好两个实验')
        return
    left, right = [mapping[i] for i in selected]
    st.table(pd.DataFrame({'参数 / 结果': [LABELS[k] for k in FIELDS] + ['创建时间 (UTC)'],
        f"#{left['id']} {left['name']}": [str(left[k]) if left[k] is not None else '尚未评测' for k in FIELDS] + [left['created_time']],
        f"#{right['id']} {right['name']}": [str(right[k]) if right[k] is not None else '尚未评测' for k in FIELDS] + [right['created_time']]}))
    chart = pd.DataFrame([{'实验': f"#{r['id']} {r['name']}", 'Success rate (%)': r['success_rate']} for r in [left, right] if r['success_rate'] is not None])
    if not chart.empty:
        st.plotly_chart(px.bar(chart, x='实验', y='Success rate (%)', range_y=[0, 100]), use_container_width=True)
    if left['success_rate'] is not None and right['success_rate'] is not None:
        st.metric('第二个实验相对第一个实验', f"{right['success_rate'] - left['success_rate']:+.2f} 个百分点")
    st.caption('请确认数据集和评测协议一致，再解释结果差异。')
