
import random
import numpy as np
import pickle
import os
import pandas as pd
from collections import deque

def create_jobs(jobs_name, num_insts, jobdir):
    array_list = []
    num_jobs = 0
    for jb, nums in zip(jobs_name, num_insts):
        m_file = os.path.join(jobdir, jb + "_m.csv")
        df = pd.read_csv(m_file, header=None)
        model_def = df.to_numpy()
        for n in range(nums):
            array_list.append(deque(model_def))
            num_jobs += len(model_def)
    array_idx_track = [0 for _ in range(len(array_list))]
    jobs = [None for _ in range(num_jobs)]
    for i in range(len(jobs)):
        arr_idx = random.choice([a_id for a_id, a in enumerate(array_list) if len(a)> 0])
        jobs[i] = (arr_idx, array_idx_track[arr_idx], array_list[arr_idx].popleft())
        array_idx_track[arr_idx] += 1
    return jobs

#============write out csv======================================================
# for n in range(10):
#     instsfile = "./batch2/insts{}.csv".format(n)
#     instsdiscription = "./batch2/description/insts{}.csv".format(n)
#     # picks = np.random.randint(0, 9, 3)
#     picks = [n]
#     jobs_name = ["traffic{}".format(picks[0]) for i in range(len(picks))]
#     # num_inst_list = [16, 16, 16]
#     num_inst_list = [1]
#     jobs = create_jobs(jobs_name, num_inst_list, "../traffic/batch2")
#     with open(instsdiscription, "w") as fd:
#         fd.write(",".join(["{}".format(a) for a in num_inst_list]))
#     with open(instsfile, "w") as fd:
#         for i in range(len(jobs)):
#             threadId, layer_Id, dim = jobs[i]
#             fd.write("{},{},{},{},{},{},{},{},{}\n".format(threadId, layer_Id, *dim))
#

#==============================================================================

#===========write out plt=========================================================
# for n in range(7, 11, 1):
#     instsfile = "insts{}.plt".format(n)
#     picks = np.random.randint(0, 9, 3)
#     jobs_name = ["traffic{}".format(picks[0]) for i in range(len(picks))]
#     num_inst_list = [16, 16, 16]
#     jobs = create_jobs(jobs_name, num_inst_list, "../traffic")
#     with open(instsfile, "wb") as fd:
#         chkpt = {"jobs":jobs,
#                  "num_inst_list":num_inst_list}
#         pickle.dump(chkpt, fd)
#================================================================================

#==================plt to csv====================================================
# for i in range(1, 11, 1):
#     instsfile = "insts{}.plt".format(i)
#     instsdes = "insts{}.csv".format(i)
#     instsdiscription = "./description/insts{}.csv".format(i)
#     chkpt = pickle.load(open(instsfile, "rb"))
#     jobs = chkpt["jobs"]
#     with open(instsdiscription, "w") as fd:
#         fd.write("{},{},{}".format(*num_inst_list))
#     with open(instsdes, "w") as fd:
#         for i in range(len(jobs)):
#             threadId, layer_Id, dim = jobs[i]
#             fd.write("{},{},{},{},{},{},{},{},{}\n".format(threadId, layer_Id,*dim))
#================================================================================


# #============write out csv blend======================================================
# for n in range(10):
#     instsfile = "./batch3/insts{}.csv".format(n)
#     instsdiscription = "./batch3/description/insts{}.csv".format(n)
#     os.makedirs("./batch3/", exist_ok=True)
#     os.makedirs("./batch3/description/", exist_ok=True)
#     jobs_name = ["./CONV/traffic{}".format(n), "./GEMM/traffic{}".format(n), "./FNN/traffic{}".format(n)]
#     num_inst_list = [1, 1, 1]
#     jobs = create_jobs(jobs_name, num_inst_list, "../traffic/batch3")
#     with open(instsdiscription, "w") as fd:
#         fd.write(",".join(["{}".format(a) for a in num_inst_list]))
#     with open(instsfile, "w") as fd:
#         for i in range(len(jobs)):
#             threadId, layer_Id, dim = jobs[i]
#             fd.write("{},{},{},{},{},{},{},{},{}\n".format(threadId, layer_Id, *dim))
#
#
# #==============================================================================







#============write out csv blend======================================================
for n in range(10):
    instsfile = "./batch_recom/insts{}.csv".format(n)
    os.makedirs("./batch_recom/", exist_ok=True)
    df = pd.read_csv("/Users/chuchu/Documents/gt_local/sepFF/fetch_model/model_def/sum/recom_net_m.csv").to_numpy()
    picks = np.random.randint(0, len(df)-1, (100,))
    pick_df = df[picks]
    num_inst_list = [1, 1, 1]
    with open(instsfile, "w") as fd:
        for i in range(len(pick_df)):
            dim = pick_df[i]
            fd.write("{},{},{},{},{},{},{},{},{}\n".format(1, 1, *dim))


#==============================================================================


