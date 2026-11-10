import json
from pathlib import Path
import pandas as pd
from tools.annotation_app import server

ROOT = Path(__file__).resolve().parents[1]
q = json.loads((ROOT / 'data/curated_cvc_v1/carina_placement_pilot_50.json').read_text())
df = pd.read_csv(ROOT / 'data/curated_cvc_v1/carina_placement_pilot_50.csv')
splits = json.loads((ROOT / 'data/splits.json').read_text())
val = set(map(str, splits['partitions']['dev_pilot_val']['study_ids']))
train = set(map(str, splits['partitions']['dev_full_train']['study_ids']))

def test_queue_is_fixed_size_and_patient_separated():
    assert len(df) == q['study_count'] == 50
    assert df.study_id.nunique() == df.patient_id.nunique() == df.duplicate_group_id.nunique() == 50
    assert set(df.study_id).issubset(train)
    assert set(df.study_id).isdisjoint(val)
    assert set(df.patient_id).isdisjoint(set(pd.read_parquet(ROOT / 'data/curated_cvc_v1/study_manifest.parquet').query("split_membership == 'dev_pilot_val'").patient_id.astype(str)))

def test_queue_strata_counts_are_exact():
    assert df.queue_stratum.value_counts().to_dict() == {'multi_cvc': 10, 'single_cluttered': 10, 'single_low_clutter': 10, 'border_or_short_trace': 10, 'placement_mix': 10}
    assert (df.loc[df.queue_stratum == 'multi_cvc', 'num_annotated_cvcs'] >= 2).all()
    assert (df.loc[df.queue_stratum == 'single_cluttered', 'num_annotated_cvcs'] == 1).all()
    assert df.loc[df.queue_stratum == 'single_cluttered', 'has_other_lines'].all()
    assert (df.loc[df.queue_stratum == 'single_low_clutter', 'num_annotated_cvcs'] == 1).all()
    assert (~df.loc[df.queue_stratum == 'single_low_clutter', 'has_other_lines']).all()
    difficult = df[df.queue_stratum == 'border_or_short_trace']
    assert (difficult.has_border_touch | difficult.min_path_length_px.lt(800)).all()

def test_annotation_contract_and_dual_reviewer_isolation():
    assert len(server.get_queue_study_ids('carina_placement_pilot_50')) == 50
    assert server._valid_native_point({'x': 0, 'y': 0}, 100, 100)
    assert not server._valid_native_point({'x': float('nan'), 'y': 2}, 100, 100)
    assert not server._valid_native_point({'x': 100, 'y': 2}, 100, 100)
    first = server._review_file('carina_placement_pilot_50', 'study', 'reviewer-a')
    second = server._review_file('carina_placement_pilot_50', 'study', 'reviewer-b')
    assert first != second
    assert first.name == second.name == 'review.json'
