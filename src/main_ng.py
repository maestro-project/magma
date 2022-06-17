
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from concurrent import futures
import nevergrad as ng
import random
import os, sys
import numpy as np
import copy
import argparse
from src.cost_model_env import MaestroEnvironment
from src.scheuler_GA import SchedulerGA
from multiprocessing.pool import ThreadPool

from datetime import datetime
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
from src.setting import  *
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []
tune_iter = 1
choose_optimizer=None
search_level = 1

def crete_random(dimension):
    indv = []
    indv.append(random.randint(0,3))
    indv.append(random.randint(0,720-1))
    for d in dimension[:-3]:
        if d ==1:
            indv.append(1)
        else:
            indv.append(random.randint(1, d))
    for d in dimension[-3:-1]:
        indv.append(random.choice(np.arange(1, d+1, 2)))
    return indv
def random_sample(num_generations=100,num_population=100,dimension=None):
    best_reward = -float("Inf")
    best_sol = None
    for g in range(num_generations):
        gen_best =  -float("Inf")
        population = [crete_random(dimension) for _ in range(num_population)]
        for i in range(len(population)):
            reward = env.oberserve_maestro(population[i])
            if reward is None:
                reward = [float("-Inf")]
            gen_best = max(gen_best, reward[0])
            gen_best_idx = i
        if best_reward < gen_best:
            best_reward = gen_best
            best_sol = copy.deepcopy(population[gen_best_idx])
        print("Gen {}: Gen reward: {}, Best reward: {}".format((g + 1), gen_best,best_reward))
        # print("Best Sol:")
        # env.print_indv(best_sol)






def eval_function(sch, *param):
    param = np.array(param).reshape(2, -1)
    return abs(sch.test([param]))



def thread_fun(sch, optimizer):
    reward = ng_search(sch, optimizer, num_generations=opt.epochs)
    return reward



def ng_search(sch,choose_optimizer, num_generations=100,num_population=1):
    param = [ng.p.Scalar(lower=0, upper=len(cores_cfg) - 1).set_integer_casting() for _ in range(len(jobs))] +  [ng.p.Scalar(lower=0, upper=1) for _ in range(len(jobs))]
    parametrization = ng.p.Instrumentation( *param)
    optimizer = ng.optimizers.registry[choose_optimizer](parametrization=parametrization, budget=num_generations* num_population, num_workers=1)
    # optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=100)
    partial_func = partial(eval_function,sch)
    recommendation = optimizer.minimize(partial_func)
    answer = recommendation.args
    reward = eval_function(sch, answer)
    print("Fitness: {:.2e}".format(reward))
    return reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default="output", help='pickle file name')
    parser.add_argument('--alg', type=str, default="PSO", help='pickle file name')
    parser.add_argument('--epochs', type=int, default=3, help='pickle file name')
    parser.add_argument('--trfdir', type=str, default="../traffic", help='pickle file name')
    parser.add_argument('--instsdir', type=str, default="batch_mix", help='pickle file name')
    parser.add_argument('--instsfile', type=str, default="insts0", help='pickle file name')
    parser.add_argument('--setting', type=int, default=0, help='pickle file name')
    parser.add_argument('--dram_bw', type=int, default=1, help='pickle file name')
    parser.add_argument('--opt_idxs', type=int, default=0, help='pickle file name')
    parser.add_argument('--df', type=str, default="eye", help='pickle file name')
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
    # os.makedirs(monitor_log, exist_ok=True)
    chkpt_file = "{}".format(now_time)
    # log_dir = os.path.join(monitor_log, chkpt_file)
    outdir_chkpt = os.path.join(outdir_exp, "chkpt")
    os.makedirs(outdir_chkpt, exist_ok=True)
    chkpt_file = os.path.join(outdir_chkpt, chkpt_file + "_c.plt")
    model_file = os.path.join(outdir_chkpt, chkpt_file + "_m.plt")
    image_before_file = os.path.join(outdir_chkpt, chkpt_file + "_b.jpg")
    image_after_file = os.path.join(outdir_chkpt, chkpt_file + "_r.jpg")
    record_table_file_r = os.path.join(outdir_chkpt, chkpt_file + "_r.csv")
    record_table_file_ng = os.path.join(outdir_chkpt, chkpt_file + "_ng.csv")
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
    insts_table_dir_NAME = "Inst-{}-{}_S-{}".format(opt.instsdir, opt.instsfile, opt.setting)
    insts_table_dir_discrption = os.path.join(insts_table_dir_discrption, insts_table_dir_NAME + ".csv")
    insts_table_file = os.path.join(insts_table_dir, insts_table_dir_NAME + ".plt")
    instsdescrpfile = os.path.join(instdir, "description/" + opt.instsfile + ".csv")

    is_random = (opt.instsfile == "random")
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
    num_cores, cores_cfg, pe_cfg, cls_cfg, style_cfg = get_setting_f(opt.setting)
    is_flex, sel_pe = is_flexible(opt.setting)

    cost_model = MaestroEnvironment(num_cores, pe_cfg, cls_cfg, style_cfg, fitness=["latency", "req_dram_bw", "l2_size"])



    optimizer_list = ["PSO", "Portfolio", "OnePlusOne","CMA", "DE","NaiveTBPSA","cGA","CauchyLHSSearch", "HaltonSearch", "HammersleySearch", "MetaRecentering"]
    # train_model(model_defs, num_generations=opt.epochs, num_population=opt.num_pop)

    if opt.setting < 4:
        dram_bw_list = [2**i for i in range(5)]
        # dram_bw_list = [1, 4, 16]
    else:
        # dram_bw_list = [2 ** i for i in range(9)]
        dram_bw_list = [1, 16, 256]
        # dram_bw_list = [1, 16, 256]
    record_table = defaultdict(list)

    #=====single thread version=================================================================
    for dram_bw in dram_bw_list:
        sch_GA = SchedulerGA(num_cores, cores_cfg, style_cfg, cost_model, jobs=jobs, tb_dir=tb_dir, \
                             chkpt_file=chkpt_file, is_random=is_random, \
                             insts_table_file=insts_table_file, true_random=False, dram_bw=dram_bw, is_flex=is_flex, sel_pe=sel_pe)
        # for opt_idx in range(7, len(optimizer_list),1):
        # for opt_idx in range(7):
        choose_optimizer = opt.alg
        reward = ng_search(sch_GA, choose_optimizer, num_generations=opt.epochs)
        record_table[choose_optimizer].append(reward)
        df = pd.DataFrame(record_table)
        df.to_csv(record_table_file_ng)
    #============================================================================================


    #======THREAD version==========================================================================
    # args = []
    # for dram_bw in dram_bw_list:
    #     sch_GA = SchedulerGA(num_cores, cores_cfg, style_cfg, cost_model, jobs=jobs, tb_dir=tb_dir, \
    #                          chkpt_file=chkpt_file, is_random=is_random, \
    #                          insts_table_file=insts_table_file, true_random=False, dram_bw=dram_bw)
    #     for opt_idx in range(len(optimizer_list)):
    #         choose_optimizer = optimizer_list[opt_idx]
    #         args.append([sch_GA, choose_optimizer])
    #
    # pool = ThreadPool(len(args))
    # reward_list = pool.starmap(thread_fun, args)
    #
    # index = 0
    # for dram_bw in dram_bw_list:
    #     for opt_idx in range(len(optimizer_list)):
    #         choose_optimizer = optimizer_list[opt_idx]
    #         record_table[choose_optimizer].append(reward_list[index])
    #         index += 1
    #
    # df = pd.DataFrame(record_table)
    # df.to_csv(record_table_file_ng)
    #============================================================================================




    # chkpt = genetic_search(num_generations=opt.epochs)
    # with open(chkpt_file, "wb") as fd:
    #     pickle.dump(chkpt, fd)
    #
    # best_reward_list = chkpt["best_reward_list"]
    # best_reward = chkpt["best_reward"]
    # best_sol = chkpt["best_sol"]
    # num_population = chkpt["num_population"]
    # num_generations = chkpt["num_generations"]
    # fitness = chkpt["fitness"]
    # print("Best  fitness :{:9e}".format(best_reward))
    # print("Best Sol:")
    # env.print_indv(best_sol)
    # with open(log_file, "w") as fd:
    #     fd.write("Layer: {}".format(dimension))
    #     fd.write("\nNum generation: {}, Num population: {}".format(num_generations, num_population))
    #     fd.write("\nBest  fitness :{:9e}".format(abs(best_reward)))
    #     print("Best Sol:")
    #     env.print_indv(best_sol,fd=fd)
    #
    # with open(sum_file, "a+") as fd:
    #     fd.write("\nExp [{}]: Best: [{}]".format(now, best_reward))
    #
    # font = {
    #     'weight': 'bold',
    #     'size': 12}
    # matplotlib.rc('font', **font)
    # fig = plt.figure(0)
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(best_reward_list)), np.abs(np.array(best_reward_list)), label="GA-df",  linewidth=5)
    # plt.figtext(0, 0, "Best fitness: {:9e}".format(abs(best_reward)))
    # plt.figtext(0, 0.05, "Layer: {}".format(dimension))
    # plt.ylabel(fitness,fontdict=font)
    # plt.xlabel('Generation #',fontdict=font)
    # plt.legend()
    # plt.show()
    # plt.savefig(img_file, dpi=300)

