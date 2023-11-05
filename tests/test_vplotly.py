import numpy as np
from scipy import stats

import vstats
import vplotly


def test_plotting_est_rejection_probabilities():
    share_n_group_1 = 0.3
    alpha_test = 0.05
    min_abs_effect = 0  # Null hypothesis holds.
    sd_1 = 8
    sd_2 = 10
    conf_level_rejection_prob = 0.95
    n_simulation = 10000

    est_welchs_test_properties = vstats.get_est_welchs_test_properties(
        n=[20, 40, 80],
        share_n_group_1=share_n_group_1,
        alpha_test=alpha_test,
        rv_1=stats.norm(loc=min_abs_effect, scale=sd_1),
        rv_2=stats.norm(loc=0, scale=sd_2),
        conf_level_rejection_prob=conf_level_rejection_prob,
        n_simulation=n_simulation,
        rng=np.random.RandomState(1653)
    )

    description = (
        "normal distributions: <br>"
        "min_abs_effect=" + str(min_abs_effect) + ", "
        "sd_1=" + str(sd_1) + ", "
        "sd_2=" + str(sd_2) + ".<br>"
        "share_n_group_1=" + str(share_n_group_1) + ", "
        "alpha_test=" + str(alpha_test)
    )

    fig = vplotly.create_figure_for_rejection_probabilities(
        title=("Estimated probabilities for Welch's "
               "test rejecting the null hypothesis"
               ),
        xaxis_title="n",
        yaxis_title="est. rejection probability",
        showlegend=True
    )

    fig = vplotly.add_est_rejection_probabilities_to_figure(
        fig=fig,
        est_test_properties=est_welchs_test_properties,
        color="blue",
        name=description
    )

    actual = {
        "array": list(fig.data[0]['error_y']["array"].round(8)),
        "arrayminus": list(fig.data[0]['error_y']["arrayminus"].round(8)),
        "x": list(fig.data[1]["x"]),
        "y": list(fig.data[1]["y"].round(8))
    }

    expected_array = [0.0045417, 0.00442227, 0.00425618]

    expected_arrayminus = [0.00427861, 0.0041573, 0.00398869]

    expected_x = [
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

    expected_y = [
        0.0522, 0.05205, 0.0519, 0.05175, 0.0516, 0.05145, 0.0513,
        0.05115, 0.051, 0.05085, 0.0507, 0.05055, 0.0504, 0.05025,
        0.0501, 0.04995, 0.0498, 0.04965, 0.0495, 0.04935, 0.0492,
        0.0491, 0.049, 0.0489, 0.0488, 0.0487, 0.0486, 0.0485,
        0.0484, 0.0483, 0.0482, 0.0481, 0.048, 0.0479, 0.0478,
        0.0477, 0.0476, 0.0475, 0.0474, 0.0473, 0.0472, 0.0471,
        0.047, 0.0469, 0.0468, 0.0467, 0.0466, 0.0465, 0.0464,
        0.0463, 0.0462, 0.0461, 0.046, 0.0459, 0.0458, 0.0457,
        0.0456, 0.0455, 0.0454, 0.0453, 0.0452
    ]

    expected = {
        "array": expected_array,
        "arrayminus": expected_arrayminus,
        "x": expected_x,
        "y": expected_y
    }

    assert actual == expected
