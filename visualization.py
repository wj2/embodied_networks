import numpy as np

import general.plotting as gpl


@gpl.ax_adder()
def plot_eye_movements(
    out_dims,
    times,
    before_time=-5,
    end_time=0,
    body_dims=(1, 2),
    ax=None,
    cmap=None,
    **kwargs,
):
    eye_moves = np.zeros((len(out_dims), -before_time + end_time, len(body_dims)))
    for i, od_trl in enumerate(out_dims):
        s_ind = times[i] + before_time
        e_ind = times[i] + end_time
        em_i = od_trl[s_ind:e_ind, body_dims]
        eye_moves[i] = em_i
        gpl.plot_colored_line(em_i[:, 0], em_i[:, 1], cmap=cmap, ax=ax)
    return eye_moves


@gpl.ax_adder()
def plot_cat_eye_movements(out_dims, times, cats, ax=None, cmaps=None, **kwargs):
    u_cats = np.unique(cats)
    if cmaps is None:
        cmaps = ("Reds", "Blues", "Greens")
    rs = np.linspace(0, np.pi * 2, 100)
    ax.plot(np.sin(rs), np.cos(rs), color="k")
    for i, uc in enumerate(u_cats):
        mask = uc == cats
        print(np.mean(mask))
        plot_eye_movements(
            out_dims[mask], times[mask].to_numpy(), ax=ax, cmap=cmaps[i], **kwargs
        )
