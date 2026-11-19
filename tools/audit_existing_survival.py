"""Independent audit of archived prediction exports; never repairs their validity."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.survival_metrics import evaluate_survival, kaplan_meier


def audit(bundle):
    predictions = pd.read_csv(bundle / 'validation_predictions.csv')
    manifest = pd.read_csv(bundle / 'data_audit/modality_manifest.csv')
    training = manifest[manifest['split'].eq('training')]
    contract = json.loads((bundle/'data_audit/feature_contract.json').read_text(encoding='utf-8-sig'))
    recorded = json.loads((bundle/'complete_metrics_summary.json').read_text(encoding='utf-8-sig'))
    forbidden = [name for name in contract['cli_columns'] if any(key in name.casefold()
        for key in ('last_contact', 'followup', 'follow_up', 'death', 'vital_status', 'survival'))]
    result = {
        'audit_scope': 'Independent arithmetic and artifact audit; archived model not retrained',
        'bundle': str(bundle.resolve()), 'predictor_leakage_candidates': forbidden,
        'publication_eligible': False,
        'reason': 'Known follow-up predictor; no clinical comparator; no usable radiology. Arithmetic agreement does not validate the model.',
        'checkpoint_hash_matches_export': hashlib.sha256((bundle/'CancerFusion_Phase2_Weights.pth').read_bytes()).hexdigest() == recorded['checkpoint_sha256'],
        'training_cases': len(training), 'training_events': int(training['event'].sum()),
        'patient_overlap': len(set(training['Patient_ID']) & set(predictions['patient_id'])),
        'duplicate_validation_ids': int(predictions['patient_id'].duplicated().sum()),
        'metrics': evaluate_survival(predictions.time, predictions.event, predictions.fusion_risk,
            training.time, training.event, horizons=[365,1095,1825]),
        'validation_observed_km': kaplan_meier(predictions.time, predictions.event),
        'modality_counts': {name: int(predictions[f'has_{name}'].sum()) for name in ('wsi','rad','rna','cli')},
        'recorded_harrell_c': recorded['all_validation_cases']['scores']['fusion_risk']['harrell_c'],
        'limitations': ['The validation set was inspected and is not an untouched final test.',
            'AUC uses risk ranking and observed outcomes under independent censoring; it does not repair clinical predictor leakage.',
            'No survival probabilities exported: no Brier/calibration/uncertainty-error evaluation is justified.',
            'No four-input validation case: four-versus-three modality advantage cannot be computed.']}
    result['c_index_matches_export'] = abs(result['metrics']['harrell_c'] - result['recorded_harrell_c']) < 1e-12
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('RUN REPORT/curated-cdr-v2'))
    parser.add_argument('--output', type=Path, default=Path('RUN REPORT/curated-cdr-v2-independent-audit.json'))
    args = parser.parse_args()
    result = audit(args.bundle)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('training_cases','training_events','patient_overlap','checkpoint_hash_matches_export','predictor_leakage_candidates','c_index_matches_export')}))
    print('Arithmetic C-index:', result['metrics']['harrell_c'], 'NOT PUBLICATION-ELIGIBLE')
