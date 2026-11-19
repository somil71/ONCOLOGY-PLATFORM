"""Hand-calculated censoring, ties, bootstrap and support regression checks."""
import json
from pathlib import Path

import numpy as np

from src.survival_metrics import (CensoringDistribution, bootstrap_concordance,
    calibration_table, evaluate_survival, evaluate_uncertainty, harrell_c, horizon_metrics, kaplan_meier, uno_c)


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected invalid data/support to be rejected")


def main():
    time = np.array([1., 2., 2., 3.])
    event = np.array([1, 1, 0, 1])
    risk = np.array([4., 3., 1., 2.])
    assert harrell_c(time, event, risk) == 1
    assert harrell_c(time, event, -risk) == 0
    assert harrell_c(time, event, np.ones(4)) == .5
    assert harrell_c([1, 2], [0, 0], [2, 1]) is None
    # Only the failure/censor tie at t=2 is comparable, not two tied failures.
    assert harrell_c([2, 2, 2], [1, 0, 1], [3, 2, 1]) == .5
    np.testing.assert_allclose(harrell_c([1, 2, 3], [1, 1, 1], [3, 2, 2]), 5/6)
    steps = kaplan_meier(time, event)
    np.testing.assert_allclose([row['survival'] for row in steps], [.75, .5, 0])
    assert [row['at_risk'] for row in steps] == [4, 3, 1]
    for row in steps:
        assert 0 <= row['lower_95'] <= row['survival'] <= row['upper_95'] <= 1
    censor = CensoringDistribution(time, event)
    np.testing.assert_allclose(censor.at([0, 1, 1.9, 2, 3]), [1, 1, 1, .5, .5])
    # A censored patient at t=2 is neither a case nor a known control at t=2.
    # Case weights 1 and 2; surviving control weight 2; (.04+.32+.18)/4=.135.
    horizon = horizon_metrics(time, event, risk, time, event, 2, [.2, .4, .8, .7])
    assert horizon['cases'] == 2 and horizon['controls'] == 1
    assert horizon['censored_by_horizon'] == 1
    assert horizon['ipcw_auc'] == 1
    np.testing.assert_allclose(horizon['ipcw_brier'], .135)
    # No censoring: IPCW Brier reduces to ordinary squared survival error.
    no_censor = horizon_metrics([1, 2, 3, 4], [1]*4, [4, 3, 2, 1],
                               [1, 2, 3, 4], [1]*4, 2.5, [.1, .2, .7, .9])
    np.testing.assert_allclose(no_censor['ipcw_brier'], (.01+.04+.09+.01)/4)
    assert uno_c(time, event, risk, time, event, 2.5) == 1
    boot = bootstrap_concordance(time, event, risk, reference_risk=risk, n_bootstrap=80)
    assert boot['paired_difference_bootstrap_95_ci'] == [0, 0]
    assert boot == bootstrap_concordance(time, event, risk, reference_risk=risk, n_bootstrap=80)
    assert bootstrap_concordance([1, 2], [0, 0], [2, 1], n_bootstrap=10)['bootstrap_95_ci'] is None
    rejects(lambda: horizon_metrics(time, event, risk, [1, 1.5], [1, 0], 2))
    rejects(lambda: horizon_metrics(time, event, risk, time, event, 3))
    rejects(lambda: harrell_c([1, 0], [1, 0], [2, 1]))
    rejects(lambda: harrell_c([1, 2], [1, 2], [2, 1]))
    rejects(lambda: harrell_c([1, 2], [1, 0], [float('nan'), 1]))
    rejects(lambda: horizon_metrics(time, event, risk, time, event, 2, [1.1]*4))
    rejects(lambda: evaluate_survival(time, event, risk, time, event, [[.2,.5]]*4, [1,2], n_bootstrap=10))
    rows = calibration_table([1, 2], [0, 0], [.2, .8], 3)
    assert all(row['km_survival'] is None for row in rows)
    result = evaluate_survival(time, event, risk, time, event,
        [[.2], [.4], [.8], [.7]], [2], n_bootstrap=30)
    assert result['horizons'][0]['status'] == 'evaluated'
    json.dumps(result, allow_nan=False)
    uncertainty = evaluate_uncertainty([1, 2, 3, 4], [1]*4, [.1, .2, .5, .2], [.01, .04, .25, .64],
        [1, 2, 3, 4], [1]*4, 2.5)
    np.testing.assert_allclose(uncertainty['ipcw_error_uncertainty_correlation'], 1)
    assert uncertainty['uncertainty_strata'][0]['ipcw_brier'] < uncertainty['uncertainty_strata'][1]['ipcw_brier']
    np.testing.assert_allclose(uncertainty['selective_prediction'][-1]['ipcw_brier'], (.01+.04+.25+.64)/4)
    print('PASS: hand-calculated Harrell ties, KM, IPCW AUC/Brier, Uno, paired bootstrap, censoring support, monotonic probabilities and input guards.')


if __name__ == '__main__':
    main()
