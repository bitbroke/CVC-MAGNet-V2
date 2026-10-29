"""Compare fixed final checkpoints and render unchanged audit panel from cached probabilities."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_cvc_masks_native import compute_patient_bootstrap_ci
from src.data.transforms import LetterboxTransform
OUT = ROOT/'runs/paired_repaired_20261016'

def main():
    reports = {'frozen_s2':json.loads((ROOT/'reports/native_eval_S2_higher_res_final.json').read_text())}
    reports['frozen_s2']['per_study_results'] = json.loads((ROOT/'reports/native_eval_S2_higher_res_final_per_study.json').read_text())
    for arm in ['s1_512','s2_768']:
        reports[arm] = json.loads((OUT/f'evaluation/native_evaluation_{arm}.json').read_text())
    target_ids = {r['study_id'] for r in reports['frozen_s2']['per_study_results']}
    assert all({r['study_id'] for r in rep['per_study_results']} == target_ids for rep in reports.values())
    reference = {r['study_id']:r for r in reports['frozen_s2']['per_study_results']}
    for rep in reports.values():
        for r in rep['per_study_results']:
            old = reference[r['study_id']]
            assert r['orig_size'] == old['orig_size']
            assert r['targ_area_native_px'] == old['targ_area_native_px']
            assert r['num_cvc_instances'] == old['num_cvc_instances']
    summaries = {k:v['summary_by_threshold']['tau_50'] for k,v in reports.items()}
    ci = {k:compute_patient_bootstrap_ci(reports['frozen_s2']['per_study_results'], reports[k]['per_study_results'], 'tau_50','tau_50') for k in ['s1_512','s2_768']}
    ledgers = {k:[json.loads(s) for s in (OUT/k/'exposure_ledger.jsonl').read_text().splitlines()] for k in ['s1_512','s2_768']}
    a,b=ledgers.values()
    same_exposure = len(a)==len(b) and all(x['study_ids']==y['study_ids'] and x['aug_params']==y['aug_params'] for x,y in zip(a,b))
    common_prefix_equal = all(x['study_ids']==y['study_ids'] and x['aug_params']==y['aug_params'] for x,y in zip(a,b))
    frozen = json.loads((ROOT/'outputs/audit50_repaired_pilot_20261016/frozen_reference_hashes.json').read_text())
    for r in frozen:
        assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['sha256']
    result={'fixed_threshold':0.5,'summaries':summaries,'paired_patient_bootstrap_vs_frozen_s2':ci,
            'identical_attempt_image_and_augmentation_sequence':same_exposure,
            'shared_attempt_prefix_identical':common_prefix_equal,
            'attempts':{k:len(v) for k,v in ledgers.items()},'frozen_hashes_unchanged':True,
            'limitations':['Development set repeatedly used; not independent test.',
                          'Only eight negative studies; specificity estimate limited.',
                          'Annotation masks are 10-native-pixel strokes, not physical silhouettes.',
                          'Historical vs repaired runs differ in more than one training detail.']}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    panel=json.loads((ROOT/'outputs/audit50_repaired_pilot_20261016/quantitative_audit.json').read_text())
    gallery=OUT/'overlays';gallery.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    cv2.setNumThreads(1)
    for item in panel:
        i=item['index'];sid=item['study_id'];im=cv2.imread(item['image_path'],0);h,w=im.shape
        # Preserve original and supplied-annotation panels exactly from the inspected sheet.
        old=Image.open(ROOT/f'outputs/audit50_repaired_pilot_20261016/case_{i:02d}.jpg')
        canvas=Image.new('RGB',(3000,665),(18,22,28));canvas.paste(old,(0,0));draw=ImageDraw.Draw(canvas)
        for j,arm in enumerate(['s1_512','s2_768']):
            rep=reports[arm];size=rep['img_size'];sha=rep['checkpoint_sha256']
            caches=list((ROOT/'cache/mask_probabilities').glob(f'native_{sha[:12]}_v3_{size}_*{sid}.npz'))
            assert caches,(arm,sid)
            with np.load(caches[-1]) as z:
                assert str(z['ckpt_sha256'])==sha
                pm=z['prob_map']
            m=LetterboxTransform((size,size)).compute_params(h,w)
            pm=cv2.resize(pm[m.pad_top:m.pad_top+m.new_h,m.pad_left:m.pad_left+m.new_w],(w,h))
            pred=pm>=.5;rgb=np.repeat(im[:,:,None],3,axis=2)
            rgb[pred]=(rgb[pred]*.4+np.array([255,70,180])*.6).astype('uint8')
            tile=Image.fromarray(rgb);tile.thumbnail((590,590));x=1800+j*600
            canvas.paste(tile,(x+(600-tile.width)//2,65))
            draw.text((x+10,34),f'Repaired {size} prediction >=0.50',font=font,fill='white')
        canvas.save(gallery/f'case_{i:02d}.jpg',quality=95)
    print('Rendered 50 paired comparisons',flush=True)

if __name__=='__main__':main()
