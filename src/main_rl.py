
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

import argparse
import copy
from cost_model_env import MaestroEnvironment
import pickle

from my_stable_baselines.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from my_stable_baselines import DDPG, TD3,SAC, PPO2, HER, DQN,ACKTR,A2C,ACER,TRPO
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
    best_reward = info[0]["best_reward"]

    return best_reward


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
    parser.add_argument('--alg', type=str, default="A2C", help='pickle file name')
    parser.add_argument('--epochs', type=int, default=3, help='pickle file name')
    parser.add_argument('--trfdir', type=str, default="../traffic", help='pickle file name')
    parser.add_argument('--instsdir', type=str, default="batch_mix", help='pickle file name')
    parser.add_argument('--instsfile', type=str, default="insts0", help='pickle file name')
    parser.add_argument('--setting', type=int, default=3, help='pickle file name')
    parser.add_argument('--dram_bw', type=int, default=1, help='pickle file name')
    parser.add_argument('--df', type=str, default="eye", help='pickle file name')
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
    exp_name = "{}_Inst-{}-{}_S-{}_BW-{}".format(opt.alg, opt.instsdir, opt.instsfile, opt.setting, opt.dram_bw)
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

    if opt.df == "eye":
        insts_table_dir = os.path.join(history_path, "traffic_table_v2")
        get_setting_f = get_setting
    elif opt.df == "shi":
        insts_table_dir = os.path.join(history_path, "traffic_table_v3")
        get_setting_f = get_setting_v3()
    else:
        print("Error!")
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
    num_cores, cores_cfg, pe_cfg, cls_cfg, style_cfg = get_setting_f(opt.setting)
    is_flex, sel_pe = is_flexible(opt.setting)
    cost_model = MaestroEnvironment(num_cores, pe_cfg, cls_cfg, style_cfg, fitness=["latency", "req_dram_bw", "l2_size"])


    #==================================================================

    over_all_best_pop = None
    if opt.setting < 4:
        dram_bw_list = [2 ** i for i in range(5)]
        # dram_bw_list = [1, 16]
    else:
        # dram_bw_list = [2 ** i for i in range(9)]
        dram_bw_list = [1, 16, 256]

    alg_list = [opt.alg]
    record_table = defaultdict(list)
    env = SchedulerRL_AB(num_cores,  cores_cfg, style_cfg, num_inst=100, cost_model=cost_model,traffic_window=5,jobs=jobs, tb_dir=tb_dir, \
                         chkpt_file=chkpt_file,
                         insts_table_file=insts_table_file,
                         algA="RL", algB="RL", dram_bw=opt.dram_bw, save_all_records=opt.save_all_records)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False)

    # generate_expert_traj(expert_greedy, "try_expert", env, n_episodes=10)

    model_def = [128,128,128]
    for alg in alg_list:
        if alg == "A2C":
            agent = A2C("MlpPolicy", env, verbose=0,  tensorboard_log=tensorboard_log,policy_kwargs=dict(layers=model_def))
        elif alg == "ACKTR":
            agent = ACKTR("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log,policy_kwargs=dict(layers=model_def))
        elif alg == "PPO2":
            agent = PPO2("MlpPolicy", env, verbose=0,  tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        elif alg=="DQN":
            agent = DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        elif alg=="TRPO":
            agent = TRPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        elif alg=="ACER":
            agent = ACER("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        elif alg=="SAC":
            agent = SAC("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        elif alg == "DDPG":
            agent = DDPG("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, policy_kwargs=dict(layers=model_def))
        agent.learn(total_timesteps=total_timesteps,callback=save_agent,
                    tb_log_name=now_time)
        best_reward = single_test(agent, env, deterministic=True)
        print("#" * 5 + "Test" + "#" * 5)
        print("{:.2e}".format(best_reward))
        record_table[alg].append(best_reward)
    df = pd.DataFrame(record_table)
    df.to_csv(record_table_file_r)
    # df.to_csv("here.csv")
    #===========================================================================================
    #
    # model_file = "../Trainedmodels/try_md"
    # agent.save(model_file)
    #
    # print("#"*5+"Test"+"#"*5)


    # #======Testing=============================================================================
    # agent.load(model_file)
    # obs = env.reset()
    # while True:
    #     action, _states = agent.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    # #=========================================================================================

