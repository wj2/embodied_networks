import numpy as np
import pandas as pd

import general.utility as u
import general.torch.rnns as grnn
import general.tasks.cognitive as gtc


def unpack_ind(start_inds, prefix):
    return u.merge_dict(start_inds, prefix=prefix + "_")


def sample_model_responses(task, model, **kwargs):
    info, inps, targs, out = grnn.sample_model_responses(task, model, **kwargs)
    start_dict = unpack_ind(info["start_ind"], "start")
    info = info.join(pd.DataFrame(start_dict))
    end_dict = unpack_ind(info["end_ind"], "end")
    info = info.join(pd.DataFrame(end_dict))
    hidden_activity = []
    out_activity = []
    for out_i, hidden_i in out:
        hidden_activity.append(hidden_i.detach().numpy())
        out_activity.append(out_i.detach().numpy())

    hidden_activity = np.array(hidden_activity, dtype=object)
    out_activity = np.array(out_activity, dtype=object)
    info["sample_cat"] = gtc.categorize(info["sample"], task.cat_dir)
    info["test_cat"] = gtc.categorize(info["sample"], task.cat_dir)
    return info, inps, targs, hidden_activity, out_activity
