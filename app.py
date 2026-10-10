"""LabTrack entry point: navigation and shared error handling."""
import logging
import sqlite3
import pandas as pd
import streamlit as st
from labtrack.database import ExperimentStore
from labtrack.ui import dashboard, experiment_form, experiment_detail, csv_management, experiment_comparison
from labtrack.validation import STATUSES

st.set_page_config(page_title='LabTrack', page_icon='🧪', layout='wide')
st.title('🧪 LabTrack')
st.caption('AI / 深度学习 / 机器人与 VLA 实验管理')


def main():
    store = ExperimentStore()
    notice = st.session_state.pop('notice', None)
    if notice:
        st.success(notice)
    page = st.sidebar.radio('导航', ['Dashboard', '实验列表', '新建实验', '实验比较', 'CSV 导入与导出'])
    if page == 'Dashboard':
        dashboard(store.list())
    elif page == '新建实验':
        st.header('新建实验')
        values = experiment_form('create')
        if values is not None:
            experiment_id = store.create(values)
            st.session_state.notice = f'实验 #{experiment_id} 已创建'
            st.rerun()
    elif page == '实验列表':
        st.header('实验列表')
        requested = st.session_state.pop('open_experiment', None)
        if requested is not None:
            # Reset filters before creating widgets so the clone stays available for editing.
            st.session_state.name_filter = ''
            st.session_state.model_filter = ''
            st.session_state.date_filter = False
            st.session_state.status_filter = '全部'
            st.session_state.tag_filter = None
            st.session_state.selected_experiment = requested
        a, b = st.columns(2)
        name = a.text_input('搜索名称', key='name_filter')
        model = b.text_input('搜索模型', key='model_filter')
        use_dates = st.checkbox('按创建日期筛选（UTC）', key='date_filter')
        start = end = None
        if use_dates:
            a, b = st.columns(2)
            start = a.date_input('开始日期')
            end = b.date_input('结束日期')
            if start > end:
                st.warning('开始日期不能晚于结束日期')
                return
        status = st.selectbox('筛选状态', ['全部'] + list(STATUSES), key='status_filter')
        tag = st.selectbox('筛选标签', [None] + [t['name'] for t in store.list_tags() if t['experiment_count']], format_func=lambda value: '全部标签' if value is None else value, key='tag_filter')
        rows = store.list(name, model, start, end, status='' if status == '全部' else status, tag=tag or '')
        st.caption(f'共 {len(rows)} 个实验')
        if not rows:
            st.info('没有匹配的实验')
            return
        st.dataframe(pd.DataFrame([dict(r, tags=' | '.join(r['tags'])) for r in rows])[['id', 'name', 'model', 'dataset', 'status', 'tags', 'success_rate', 'created_time']], use_container_width=True, hide_index=True)
        selected = st.selectbox('选择实验查看详情', key='selected_experiment', options=[r['id'] for r in rows], format_func=lambda i: next(f"#{r['id']} · {r['name']}" for r in rows if r['id'] == i))
        row = store.get(selected)
        if row is None:
            st.warning('该实验已被删除，请刷新页面')
            return
        experiment_detail(store, row)
    elif page == 'CSV 导入与导出':
        csv_management(store)
    else:
        experiment_comparison(store)


try:
    main()
except ValueError as exc:
    st.error(str(exc))
except (sqlite3.Error, OSError):
    logging.exception('Database operation failed')
    st.error('数据库读写失败，请检查数据库路径、文件权限和磁盘空间，然后重试。详细错误已记录在终端。')
