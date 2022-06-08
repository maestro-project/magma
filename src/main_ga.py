
from math import ceil
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

import argparse
import copy
from cost_model_env import MaestroEnvironment
import pickle

# from my_stable_baselines.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
# from my_stable_baselines import DDPG, TD3,SAC, PPO2, HER, DQN,ACKTR,A2C,ACER,TRPO
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
from scheuler_rl_AB import SchedulerRL_AB
from scheuler_GA import SchedulerGA
import pandas as pd
import numpy as np
from utils import get_CONVtype_choices
from collections import  defaultdict
from setting import  *
# from my_stable_baselines.gail import generate_expert_traj
os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

n_steps = 0
def single_test(agent, env, deterministic=False):
    obs = env.reset()
    while 1:
        action, _states = agent.predict(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(action)
        if dones:
            break
    return rewards


def expert_greedy(_obs):
    return [env.action_space.sample()]

def save_agent(_locals, _globals):
    global n_steps
    if (n_steps + 1) % 200 == 0:
        agent.save(model_file)
    n_steps += 1
    return True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', type=str, default="output", help='pickle file name')
    parser.add_argument('--alg', type=str, default="GA", help='pickle file name')
    parser.add_argument('--epochs', type=int, default=2, help='pickle file name')
    parser.add_argument('--trfdir', type=str, default="../traffic", help='pickle file name')
    parser.add_argument('--instsdir', type=str, default="batch_mix", help='pickle file name')
    parser.add_argument('--instsfile', type=str, default="insts0", help='pickle file name')
    parser.add_argument('--setting', type=int, default=3, help='pickle file name')
    parser.add_argument('--dram_bw', type=int, default=1, help='pickle file name')
    parser.add_argument('--num_microBatch', type=int, default=1, help='pickle file name')
    parser.add_argument('--save_all_records', action='store_true')
    opt = parser.parse_args()
    history_path = os.path.abspath('./')
    now = datetime.now()
    now_date = "{}".format(now.date())
    now_time = "{}".format(now.time())
    outdir = opt.outdir
    outdir = os.path.join(history_path, outdir)
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(history_path, "tb_all")
    os.makedirs(tb_dir, exist_ok=True)
    tb_dir = os.path.join(tb_dir, opt.outdir)
    os.makedirs(tb_dir, exist_ok=True)
    outdir = os.path.join(outdir, now_date)
    instdir = "../traffic_insts/"
    instdir = os.path.join(instdir, opt.instsdir)
    exp_name = "{}_Inst-{}-{}_S-{}_BW-{}_NmB-{}".format(opt.alg, opt.instsdir, opt.instsfile, opt.setting, opt.dram_bw, opt.num_microBatch)
    exp_dir_name = "Inst-{}-{}_S-{}_BW-{}".format(opt.instsdir, opt.instsfile, opt.setting, opt.dram_bw)
    tb_dir = os.path.join(tb_dir, exp_dir_name)
    os.makedirs(tb_dir, exist_ok=True)
    tb_dir = os.path.join(tb_dir, exp_name)
    os.makedirs(tb_dir, exist_ok=True)
    outdir_exp = os.path.join(outdir, exp_dir_name)
    os.makedirs(outdir_exp, exist_ok=True)
    outdir_exp = os.path.join(outdir_exp, exp_name)
    tensorboard_log = os.path.join(outdir_exp, "tb_log")
    monitor_log = os.path.join(outdir_exp, "monitor")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)
    os.makedirs(monitor_log, exist_ok=True)
    chkpt_file = "{}".format(now_time)
    log_dir = os.path.join(monitor_log, chkpt_file)
    outdir_chkpt = os.path.join(outdir_exp, "chkpt")
    os.makedirs(outdir_chkpt, exist_ok=True)
    chkpt_file = os.path.join(outdir_chkpt, chkpt_file + "_c.plt")
    model_file = os.path.join(outdir_chkpt, chkpt_file + "_m.plt")
    image_file = os.path.join(outdir_chkpt, chkpt_file +"_fig_")
    record_table_file_r = os.path.join(outdir_chkpt, chkpt_file + "_r.csv")
    record_table_file_t = os.path.join(outdir_chkpt, chkpt_file + "_t.csv")
    instsfile = os.path.join(instdir, opt.instsfile + ".csv")

    insts_table_dir = os.path.join(history_path, "traffic_table_v3")
    os.makedirs(insts_table_dir, exist_ok=True)
    insts_table_dir_discrption = os.path.join(insts_table_dir, "discrption")
    os.makedirs(insts_table_dir_discrption, exist_ok=True)
    insts_table_dir_NAME ="Inst-{}-{}_S-{}".format(opt.instsdir, opt.instsfile, opt.setting)
    insts_table_dir_discrption = os.path.join(insts_table_dir_discrption, insts_table_dir_NAME + ".csv")
    insts_table_file = os.path.join(insts_table_dir, insts_table_dir_NAME+".plt")
    instsdescrpfile = os.path.join(instdir, "description/" + opt.instsfile + ".csv")

    is_random = (opt.instsfile=="random")
    # ==========Read traffic=========================================
    if not is_random:
        with open(instsfile, "r") as fd:
            df = pd.read_csv(instsfile, header=None)
            df = df.to_numpy()
            jobs = []
            for row in df:
                threadId, layerId, dim = row[0], row[1], np.array(row[2:])
                jobs.append([threadId, layerId, dim])
        jobs_backup = copy.deepcopy(jobs)
        total_timesteps = len(jobs) * opt.epochs
    else:
        jobs = []
        total_timesteps = opt.traffic_len * opt.epochs
    # ==================================================================
    #====Initial environement==========================================
    num_cores, cores_cfg, pe_cfg, cls_cfg, style_cfg = get_setting_v3(opt.setting)
    is_flex, sel_pe = is_flexible(opt.setting)
    cost_model = MaestroEnvironment(num_cores, pe_cfg, cls_cfg, style_cfg, fitness=["latency", "req_dram_bw"])

    sch_GA = SchedulerGA(num_cores,  cores_cfg, style_cfg, cost_model,jobs=jobs, tb_dir=tb_dir, \
                         chkpt_file=chkpt_file, is_random=is_random, \
                         insts_table_file=insts_table_file,true_random=False, dram_bw= opt.dram_bw, is_flex=is_flex, sel_pe=sel_pe, save_all_records=opt.save_all_records)
    sch_GA.export_job_table(insts_table_dir_discrption)
    #==================================================================

    #=======GA learning======================================================

    if opt.alg == 'GA':
        over_all_best_pop = None

        dram_bw_list = [opt.dram_bw]
        record_table = defaultdict(list)
        micro_batch_size = ceil(len(jobs)/opt.num_microBatch)
        reward_list = []
        for job_id in range(0, len(jobs), micro_batch_size):
            micro_jobs = jobs[job_id:job_id+micro_batch_size]
            sch_GA = SchedulerGA(num_cores, cores_cfg, style_cfg, cost_model, jobs=micro_jobs, tb_dir=tb_dir, \
                                 chkpt_file=chkpt_file, is_random=is_random, \
                                 insts_table_file=insts_table_file, true_random=False, dram_bw=opt.dram_bw, is_flex=is_flex, sel_pe=sel_pe, save_all_records=opt.save_all_records)
            over_all_best_reward = float("Inf")
            reward_list = []
            pops, reward = sch_GA.run(num_gen=opt.epochs, image_file=image_file, log_info=1,num_pop=len(micro_jobs))
            reward = abs(reward)
            reward_list.append(reward)
        record_table["GA"].append(np.sum(reward_list))
        df = pd.DataFrame(record_table)
        df.to_csv(record_table_file_r)

        with open(chkpt_file, 'rb') as fd:
            chkpt = pickle.load(fd)
            util = chkpt['best_utils'][-1]
            print(chkpt_file, util)

    #=======Collect random sampling data ======================================================
    elif opt.alg == 'collect_data':
        over_all_best_pop = None

        dram_bw_list = [opt.dram_bw]
        record_table = defaultdict(list)
        micro_batch_size = ceil(len(jobs)/opt.num_microBatch)
        reward_list = []
        for job_id in range(0, len(jobs), micro_batch_size):
            micro_jobs = jobs[job_id:job_id+micro_batch_size]
            sch_GA = SchedulerGA(num_cores, cores_cfg, style_cfg, cost_model, jobs=micro_jobs, tb_dir=tb_dir, \
                                 chkpt_file=chkpt_file, is_random=is_random, \
                                 insts_table_file=insts_table_file, true_random=False, dram_bw=opt.dram_bw, is_flex=is_flex, sel_pe=sel_pe, save_all_records=opt.save_all_records)
            over_all_best_reward = float("Inf")
            reward_list = []
            pops, reward = sch_GA.collect_random_sampling_data(num_gen=opt.epochs, image_file=image_file, log_info=1,num_pop=len(micro_jobs))
            reward = abs(reward)
            reward_list.append(reward)
        record_table["GA"].append(np.sum(reward_list))
        df = pd.DataFrame(record_table)
        df.to_csv(record_table_file_r)


