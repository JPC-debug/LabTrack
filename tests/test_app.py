from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_app_flow(tmp_path, monkeypatch):
    monkeypatch.setenv('LABTRACK_DB', str(tmp_path / 'ui.db'))
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not at.exception
    at.sidebar.radio[0].set_value('新建实验').run()
    at.text_input[0].set_value('UI experiment')
    at.text_input[1].set_value('Model A')
    at.text_input[2].set_value('Dataset A')
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) == 1
    at.sidebar.radio[0].set_value('实验列表').run()
    assert not at.exception
    assert at.selectbox(key='selected_experiment').value == 1
    # Edit form: search inputs precede experiment fields.
    at.text_input[2].set_value('Edited experiment')
    at.button[0].click().run()
    assert not at.exception
    assert 'Edited experiment' in at.subheader[0].value
    at.sidebar.radio[0].set_value('新建实验').run()
    at.text_input[0].set_value('Second experiment')
    at.text_input[1].set_value('Model B')
    at.text_input[2].set_value('Dataset B')
    at.checkbox[0].set_value(True)
    at.button[0].click().run()
    at.sidebar.radio[0].set_value('实验比较').run()
    at.multiselect[0].set_value([1, 2]).run()
    assert not at.exception
    assert len(at.table) == 1
    at.sidebar.radio[0].set_value('Dashboard').run()
    assert not at.exception
    assert at.metric[0].value == '2'
    at.sidebar.radio[0].set_value('实验列表').run()
    at.checkbox[-1].set_value(True).run()
    at.button[-1].click().run()
    assert not at.exception
    assert len(at.success) == 1


def test_log_import_and_metrics_ui(tmp_path, monkeypatch):
    from io import BytesIO
    import streamlit as st
    from labtrack.database import ExperimentStore
    from tests.test_metrics import create_experiment

    db = tmp_path / 'metrics_ui.db'
    monkeypatch.setenv('LABTRACK_DB', str(db))
    store = ExperimentStore(db)
    uid = create_experiment(store)
    upload = BytesIO(b'Epoch 1 loss=1 LR=.001 accuracy=.8 success rate=70%\nEpoch 2 loss=.5 LR=.0001 accuracy=.9 success rate=85%')
    upload.name = 'training.log'
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: upload)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.sidebar.radio[0].set_value('实验列表').run()
    assert not at.exception
    assert len(at.dataframe) == 2  # Experiment list and parsed preview.
    assert store.get_metrics(uid) == []  # Preview must not write anything.
    at.button(key=f'import_{uid}').click().run()
    assert not at.exception
    assert len(store.get_metrics(uid)) == 8
    assert len(at.get('plotly_chart')) == 4
    labels = {m.label: m.value for m in at.metric}
    assert labels['Final Loss'] == '0.5'
    assert labels['Best Loss'] == '0.5'
    assert labels['Final Accuracy (%)'] == '90'
    assert labels['Final Success Rate (%)'] == '85'
    assert len(at.get('download_button')) == 1
    # Re-importing the same file is idempotent.
    at.button(key=f'import_{uid}').click().run()
    assert len(store.get_metrics(uid)) == 8
    at.checkbox(key=f'clear_confirm_{uid}').set_value(True).run()
    at.button(key=f'clear_{uid}').click().run()
    assert not at.exception
    assert store.get_metrics(uid) == []


def test_invalid_log_does_not_break_details(tmp_path, monkeypatch):
    from io import BytesIO
    import streamlit as st
    from labtrack.database import ExperimentStore
    from tests.test_metrics import create_experiment

    db = tmp_path / 'invalid_ui.db'
    monkeypatch.setenv('LABTRACK_DB', str(db))
    store = ExperimentStore(db)
    create_experiment(store)
    upload = BytesIO(b'not a training log')
    upload.name = 'invalid.txt'
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: upload)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.sidebar.radio[0].set_value('实验列表').run()
    assert not at.exception
    assert any('没有解析出指标' in w.value for w in at.warning)
    assert len(at.get('download_button')) == 1
    upload = BytesIO(b'\xff')
    upload.name = 'invalid.log'
    at.run()
    assert not at.exception
    assert any('UTF-8' in e.value for e in at.error)


def test_status_tags_filters_and_command_ui(tmp_path, monkeypatch):
    from labtrack.database import ExperimentStore
    monkeypatch.setenv('LABTRACK_DB', str(tmp_path / 'third_ui.db'))
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.sidebar.radio[0].set_value('新建实验').run()
    assert at.selectbox(key='status_create').value == 'planned'
    at.text_input[0].set_value('Tagged experiment')
    at.text_input[1].set_value('OpenVLA')
    at.text_input[2].set_value('LIBERO')
    at.selectbox(key='status_create').set_value('running')
    at.text_area(key='tags_create').set_value(' VLA, VLA, baseline, ,LIBERO')
    for area in at.text_area:
        if area.label == 'Train command':
            area.set_value('python train.py --batch_size 64 --lr 0.0001 --epochs=10 --fp16')
    at.button[0].click().run()
    store = ExperimentStore(tmp_path / 'third_ui.db')
    assert store.get(1)['tags'] == ['LIBERO', 'VLA', 'baseline']
    assert store.get(1)['status'] == 'running'
    at.sidebar.radio[0].set_value('实验列表').run()
    assert not at.exception
    assert any('Batch size 不一致' in w.value for w in at.warning)
    at.selectbox(key='status_filter').set_value('completed').run()
    assert '没有匹配的实验' in at.info[0].value
    at.selectbox(key='status_filter').set_value('running').run()
    at.selectbox(key='tag_filter').set_value('baseline').run()
    assert at.selectbox(key='selected_experiment').value == 1
    # Keep the edited record visible while changing its filterable properties.
    at.selectbox(key='status_filter').set_value('全部').run()
    at.selectbox(key='tag_filter').set_value(None).run()
    at.selectbox(key='status_edit_1').set_value('completed')
    at.text_area(key='tags_edit_1').set_value('VLA\n全部')
    at.button[0].click().run()
    assert store.get(1)['status'] == 'completed'
    assert store.get(1)['tags'] == ['VLA', '全部']
    at.selectbox(key='status_filter').set_value('全部').run()
    at.selectbox(key='tag_filter').set_value('全部').run()
    assert at.selectbox(key='selected_experiment').value == 1
    at.sidebar.radio[0].set_value('Dashboard').run()
    values = {m.label: m.value for m in at.metric}
    assert values['Completed'] == '1' and values['Running'] == '0'
    assert not at.exception


def test_csv_import_ui_validation_and_confirmation(tmp_path, monkeypatch):
    from io import BytesIO
    import streamlit as st
    from labtrack.database import ExperimentStore
    monkeypatch.setenv('LABTRACK_DB', str(tmp_path / 'csv_ui.db'))
    header = b'name,model,dataset,batch_size,learning_rate,epochs,status,tags\n'
    valid = header + b'csv experiment,VLA,LIBERO,16,.001,8,running,VLA|baseline\n'
    upload = BytesIO(valid)
    upload.name = 'experiments.csv'
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: upload)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.sidebar.radio[0].set_value('CSV 导入与导出').run()
    store = ExperimentStore(tmp_path / 'csv_ui.db')
    assert not at.exception
    assert len(at.get('download_button')) == 2
    assert store.list() == []
    assert at.button(key='csv_import').disabled
    at.checkbox[0].set_value(True).run()
    at.button(key='csv_import').click().run()
    assert not at.exception and len(store.list()) == 1
    assert store.list()[0]['tags'] == ['VLA', 'baseline']
    upload = BytesIO(valid + b'bad,VLA,LIBERO,16,nan,8,unknown,debug\n')
    upload.name = 'invalid.csv'
    at.run()
    assert not at.exception
    assert any('整个文件不会写入任何数据' in e.value for e in at.error)
    assert any('第 3 行：learning_rate' in text.value for text in at.text)
    assert any('第 3 行：status' in text.value for text in at.text)
    assert len(at.button) == 0
    assert len(store.list()) == 1


def test_clone_tabs_and_followup_edit(tmp_path, monkeypatch):
    from labtrack.database import ExperimentStore
    from labtrack.parser import parse_log
    from tests.test_metrics import create_experiment
    path = tmp_path / 'clone_ui.db'
    monkeypatch.setenv('LABTRACK_DB', str(path))
    store = ExperimentStore(path)
    uid = create_experiment(store)
    store.update(uid, {'status': 'completed', 'tags': ['VLA']})
    store.add_metrics(uid, parse_log('Epoch 1 loss=.4').metrics)
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.sidebar.radio[0].set_value('实验列表').run()
    assert [tab.label for tab in at.tabs] == ['Overview', 'Metrics', 'Commands', 'Data']
    at.text_input(key='name_filter').set_value('metrics experiment').run()
    at.selectbox(key='status_filter').set_value('completed').run()
    at.button(key=f'clone_{uid}').click().run()
    assert not at.exception
    assert at.selectbox(key='selected_experiment').value == 2
    assert at.selectbox(key='status_filter').value == '全部'
    assert at.text_input(key='name_filter').value == ''
    assert store.get(2)['status'] == 'planned' and store.get(2)['success_rate'] is None
    assert store.get_metrics(2) == []
    at.selectbox(key='status_edit_2').set_value('running')
    # Existing search inputs precede the editable overview fields.
    at.text_input[2].set_value('New run from clone')
    at.button[0].click().run()
    assert not at.exception
    assert store.get(2)['name'] == 'New run from clone'
    assert store.get(2)['status'] == 'running'
    assert store.get(uid)['status'] == 'completed'
