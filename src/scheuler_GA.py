
import numpy as np
import gym
from gym import spaces
import os, datetime
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
from scipy.stats import entropy
import random
from src.utils import *
import copy
from math import floor
import pandas as pd
from collections import defaultdict
from math import ceil
style_dict = {"dla": 0, "shi": 1, "eye": 2}
MAX_styles = len(style_dict)
DIM_MAX = [256, 256, 225, 225, 3, 3, 2]
MAX_pe_cfg = 256

USE_FULL_OBSERVE=True

class SchedulerGA(object):
    def __init__(self, num_cores,  cores_cfg, style_cfg,  cost_model,  jobs=None,tb_dir="./", chkpt_file="try.plt",
                 is_random=False, insts_table_file=None, random_table_dir=None, true_random=False, dram_bw= 1,
                 is_flex=False,sel_pe=False, save_all_records=False):
        super(SchedulerGA, self).__init__()
        self.is_random = is_random
        self.core_cfg = cores_cfg
        self.style_cfg = np.array([style_dict[s] for s in style_cfg], dtype=float)
        self.num_cores = num_cores
        self.total_jobs = len(jobs) if not is_random else self.traffic_len
        self.jobs = jobs
        self.curTime = 0
        self.dram_bw = dram_bw
        self.len_dim = 6
        self.num_CONVtype = 4

        # self.state_size = len(self.core_cfg) * 2 + len_dim
        # self.action_space = spaces.MultiDiscrete([traffic_window, len(cores_cfg)])

        self.save_all_records = save_all_records
        self.cost_model = cost_model
        self.total_idle_time = 0
        self.curStep = 0
        self.epoch = 0
        self.reward_rec = []
        self.best_reward = []
        self.best_rewards = []
        self.best_utils = []
        self.best_cands = []
        self.all_records = []
        self.chkpt_file=chkpt_file

        self.job_table = np.zeros((len(self.jobs), len(self.core_cfg)))
        self.insts_table_file = insts_table_file
        self.random_table_dir = random_table_dir
        self.is_flex = is_flex
        self.is_flex_sel_pe = sel_pe
        self.true_random = true_random
        self.create_job_table()
        self.sorting_job()


    def create_pop(self):
        cores_genes = np.random.randint(0, len(self.core_cfg) , (1, len(self.job_table)))
        order_genes = np.random.random((1, len(self.job_table)))
        return np.concatenate((cores_genes, order_genes), axis=0)

    def sorting_job(self):
        sorted_array = sorted(((i, j) for j, i in enumerate(self.job_table)), key=lambda job: ([job[0][0][-i] for i in range(1, len(job[0][0])-1, 1)]))
        self.job_table = np.array([ele[0] for ele in sorted_array])
        self.job_table_index = np.array(ele[1] for ele in sorted_array)

    def sorting_corejob(self, coreJobs):
        new_coreJob = []
        for i in range(len(coreJobs)):
            curcorejob = sorted(coreJobs[i], key=lambda job: job[0])
            curcorejob_list = []
            for curjob in curcorejob:
                job_id, latency, req_dram_bw = curjob[1:]
                left_auc = latency * req_dram_bw
                curcorejob_list.append([left_auc, req_dram_bw, job_id])
            new_coreJob.append(curcorejob_list)
        return new_coreJob
    def run(self, num_gen=1000, num_pop=100, elite_ratio=0.05, parent_ratio = 0.1, pops= [], resume=False, image_file=None, log_info=2):
        # self.create_job_table(num_jobs_train=50)
        self.reset()
        if resume:
            self.pops = pops
            self.num_parent = max(2, int(num_pop * parent_ratio))
        else:
            self.pops = [self.create_pop() for _ in range(num_pop)]
            self.num_parent = num_pop

        self.num_pop = num_pop
        self.num_elite = max(1, int(num_pop* elite_ratio))
        self.fitness = np.ones((len(self.pops),))
        self.latencys = np.ones((len(self.pops),))
        self.utils = np.ones((len(self.pops),))
        self.timeTracks = np.zeros((len(self.pops), self.num_cores))
        if resume:
            self.evaluate()
        for g in range(num_gen):
            self.sel_parents()
            self.crossover(0.9)
            self.crossover_rg(0.1)
            self.crossover_acc(0.1)
            #
            # self.mutate_point_corssover()
            # self.mutate_resched(0.05)
            # self.mutate_offload(0.5)
            self.mutate(0.05)

            self.pops =  self.elite + self.pops
            self.fitness = np.concatenate((self.elite_fitness, self.fitness))
            self.latencys = np.concatenate((self.elite_latencys, self.latencys))
            self.utils = np.concatenate((self.elite_util, self.utils))
            self.timeTracks = np.concatenate((self.elite_timeTracks, self.timeTracks))
            self.evaluate()
            best_idx = np.argmax(self.fitness)
            if (log_info ==1 and (g==0 or g==num_gen-1)) or log_info>=2:
                print("[Gen {}] Fitness: Latency {:.2e}, Util: {:.2f}% ".format(g+1,self.latencys[best_idx],self.utils[best_idx] * 100))
            # print(self.pops[np.argmax(self.fitness)][0])
            self.num_parent = max(2, int(num_pop * parent_ratio))
            self.update_best_reward_list(g, best_cand = self.pops[best_idx], pops=self.pops, fitness=self.fitness)
        #     if g==0 :
        #         self.plot_pop(self.pops[0],self.fitness[0], image_file=image_file +"_b")
        # best_pop_id = np.argmax(self.fitness)
        # self.plot_pop(self.pops[best_pop_id], self.fitness[best_pop_id], image_file=image_file +"_r")
        self.save_chkpt()
        return self.pops, self.best_reward


    def collect_random_sampling_data(self, num_gen=1000, num_pop=100, elite_ratio=0.05, parent_ratio = 0.1, pops= [], resume=False, image_file=None, log_info=2):
        self.num_pop = num_pop
        self.pops = [self.create_pop() for _ in range(num_pop)]
        self.num_elite = max(1, int(num_pop* elite_ratio))
        self.fitness = np.ones((len(self.pops),))
        self.latencys = np.ones((len(self.pops),))
        self.utils = np.ones((len(self.pops),))
        self.timeTracks = np.zeros((len(self.pops), self.num_cores))
        for g in range(num_gen):
            self.pops = [self.create_pop() for _ in range(num_pop)]
            self.evaluate()
            self.num_parent = max(2, int(num_pop * parent_ratio))
            self.update_best_reward_list(g, best_cand = self.pops[0], pops=self.pops, fitness=self.fitness)
            self.save_chkpt() if self.save_all_records else None
        return self.pops, self.best_reward


    def get_best_pop_record(self):
        return self.best_rewards

    def get_tensor_type(self):
        tensor_type = []
        for job in self.jobs:
            tensor_type.append(job[-1][-1])
        return tensor_type

    def get_color(self, tensor_type, id):
        color = min(1, (id *5)%(len(tensor_type)) / len(tensor_type))
        colors = [0, 0, 0, 1]
        colors[tensor_type[id]]=color
        return colors

    def plot_pop(self, pop, max_span,image_file=None ):
        from matplotlib.patches import Rectangle
        import matplotlib
        import matplotlib.pyplot as plt
        # from matplotlib.collections import PatchCollection
        tensor_type = self.get_tensor_type()
        tensor_type = np.array(tensor_type)
        tensor_type[tensor_type==2] = 1
        prev_id = np.zeros((len(self.core_cfg),))
        prev_id[:]=-1
        font = {
            'weight': 'bold',
            'size': 20}
        matplotlib.rc('font', **font)
        matplotlib.rcParams.update({'figure.autolayout': True})
        self.get_core_stat(pop)
        fig, ax = plt.subplots()
        fig_box, ax_box = plt.subplots()
        box_y = np.arange(1, len(self.core_cfg)+1, 1)
        box_h = 0.2
        colors = ["r", "g", "y","b", "purple", "orange", "c", "darkblue", "violet", "olive", "silver", "gold", "aqua", "indigo", "sienna", "skyblue"]
        hatchs= ['/', ' \ ',  ' - ', ' + ', 'x', 'o', 'O', '.', ' * ',' | ']
        plot_x,  plot_h, plot_w, plot_job_id = self.post_process(maxBW=self.dram_bw, plot=True)
        ax.set_xlim(0, np.abs(max_span))
        ax.set_ylim(0, np.max(plot_h))
        # ax.set_xlim(np.abs(max_span)*9/10, np.abs(max_span))
        # ax.set_ylim(0, 0.0000025)
        ax_box.set_xlim(0, np.abs(max_span))
        ax_box.set_ylim(0, len(self.core_cfg)+1)
        for x,  w, h_list, id_list in zip(plot_x,  plot_w, plot_h, plot_job_id):
            prev_h =0

            for i, (h, id) in enumerate(zip(h_list,id_list)):
                rect = Rectangle((x , prev_h), w, h-prev_h, color=colors[i%len(h_list)], fill=True, hatch=hatchs[i%len(h_list)],edgecolor="k", label="Accl-{}".format(i))
                if id == prev_id[i]:
                    rect_box = Rectangle((x, box_y[i]), w, box_h, color=colors[tensor_type[id]], fill=True)
                else:
                    rect_box = Rectangle((x , box_y[i]), w, box_h, facecolor=colors[tensor_type[id]], fill=True,edgecolor="gold",linestyle="-")
                prev_h = h

                ax.add_patch(rect)
                ax_box.add_patch(rect_box)
            prev_id[:] = id_list
            handles, labels = ax.get_legend_handles_labels()
            #===TO plot legend only===============
            # ax.legend(handles[::-1], labels[::-1],loc="lower right", bbox_to_anchor=(1., 1.02) , borderaxespad=0., ncol=1)
            # ax_box.legend(handles[::-1], labels[::-1], loc='upper right')
            #
            # plt.show()
            #=================================
        # ax.text(1,1, "Makespan: {:.2e}".format(abs(max_span)),transform=ax.transAxes,horizontalalignment='right', verticalalignment='top',  fontsize=14)
        ax_box.text(1,1, "Makespan: {:.2e}".format(abs(max_span)),transform=ax_box.transAxes,horizontalalignment='right', verticalalignment='top',  fontsize=20)
        ax.set_xlabel("Time", fontdict=font)
        ax.set_ylabel("BW",fontdict=font)
        ax_box.set_xlabel("Time", fontdict=font)
        ax_box.set_ylabel("Accel", fontdict=font)
        ax_box.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.gcf().subplots_adjust(bottom=0.15)
        fig.savefig(image_file + "_bw.jpg", dpi=600) if image_file else None
        # plt.gcf().subplots_adjust(bottom=0.15)
        fig_box.savefig(image_file + "_box.jpg", dpi=600) if image_file else None
        plt.show()

    def get_best_pop(self):
        idx = np.argmax(self.fitness)
        return self.pops[idx]
    def test(self, pops, num_jobs=False):
        # self.create_job_table()
        # # self.create_job_table(num_jobs_train=50, num_jobs_test=50)
        # self.sorting_job()
        self.fitness = np.zeros((len(pops),))
        self.fitness = np.ones((len(pops),))
        self.latencys = np.ones((len(pops),))
        self.utils = np.ones((len(pops),))
        self.timeTracks = np.zeros((len(pops), self.num_cores))
        self.pops = pops
        self.evaluate()

        # print("Testing Fitness: {:.2e}".format(np.max(self.fitness)))
        return  np.max(self.fitness)

    def plot_from_chkpt(self,chkpt, image_file=None):
        self.load_chkpt(chkpt)
        self.fitness = np.zeros((len(self.pops),))
        self.fitness = np.ones((len(self.pops),))
        self.latencys = np.ones((len(self.pops),))
        self.utils = np.ones((len(self.pops),))
        self.timeTracks = np.zeros((len(self.pops), self.num_cores))
        self.evaluate()
        best_pop_id = np.argmax(self.fitness)
        self.plot_pop(self.pops[best_pop_id], self.fitness[best_pop_id],image_file=image_file + "_r")
        # print("Testing Fitness: {:.2e}".format(np.max(self.fitness)))

        return np.max(self.fitness)

    def test_raw(self, num_pop=100):
        self.create_job_table()
        self.pops = [self.create_pop() for _ in range(num_pop)]
        self.fitness = np.zeros((num_pop,))
        self.evaluate()
        print("Testing Fitness: {:.2e}".format(np.mean(self.fitness)))
        return  np.mean(self.fitness)





    def sel_parents(self):
        idx = np.argsort(self.fitness)[::-1]
        self.pops = [self.pops[i] for i in idx][:self.num_pop]
        self.fitness = self.fitness[idx][:self.num_pop]
        self.latencys = self.latencys[idx][:self.num_pop]
        self.utils = self.utils[idx][:self.num_pop]
        self.timeTracks = self.timeTracks[idx][:self.num_pop]
        self.parents = copy.deepcopy(self.pops[:self.num_parent])
        self.elite = copy.deepcopy(self.pops[:self.num_elite])
        self.elite_fitness = copy.deepcopy(self.fitness[:self.num_elite])
        self.elite_latencys = copy.deepcopy(self.latencys[:self.num_elite])
        self.elite_util = copy.deepcopy(self.utils[:self.num_elite])
        self.elite_timeTracks = copy.deepcopy(self.timeTracks[:self.num_elite])
        # weight = np.array([i/(i+1)  for i in range(len(self.parents))])
        # weight = np.array([i  for i in range(len(self.parents))])
        weight = np.array([-self.fitness[i] for i in range(len(self.parents))])
        # weight = np.array([- 1 / self.fitness[i] for i in range(len(self.parents))])
        weight = np.max(weight) - weight + 1
        weight = weight / np.sum(weight)
        self.weight = weight

    def mutate_offload(self, alhpa=0.05, off_load_ratio = 0.01):
        self.evaluate()
        for i in range(len(self.pops)):
            weight = self.timeTracks[i] + 1
            choiceL = np.argsort(weight)[::-1]
            num_cores = len(choiceL)
            off_loading_cores = ceil(num_cores * off_load_ratio)
            for k in range(len(self.pops[0][0])):
                if random.random() < alhpa:
                    if any([self.pops[i][0, k]==choice for choice in choiceL[:off_loading_cores]]):
                        self.pops[i][0][k] = np.random.choice(choiceL[-off_loading_cores:])
                        break



    def mutate(self, alhpa=0.05):
        for i in range(len(self.pops)):
            for r in range(len(self.pops[0])):
                for t in range(len(self.pops[0][0])):
                    if random.random()< alhpa:
                        if r ==0:
                            self.pops[i][r][t] = random.randint(0, len(self.core_cfg)-1)
                        else:
                            self.pops[i][r][t] = random.random()



    def mutate_point_corssover(self, alpha = 0.05):
        for idx in range(len(self.pops)):
            if random.random() < alpha:
                dad = self.pops[idx]
                pick1 = random.randint(0, len(dad[0])-1)
                pick2 = random.randint(0, len(dad[0])-1)
                dad[0,pick1], dad[0, pick2] = dad[0, pick1], dad[0,pick2]

    def mutate_resched(self, alpha = 0.1):
        for idx in range(len(self.pops)):
            if random.random() < alpha:
                dad = self.pops[idx]
                pick1 = random.randint(0, len(self.core_cfg) - 1)
                pick2 = random.randint(0, len(self.core_cfg) - 1)
                for i in range(len(dad[0])):
                   if dad[0][i]==pick1:
                       dad[0][i]= pick2
                   elif dad[0][i]==pick2:
                        dad[0][i]= pick1
    def crossover_acc(self, alpha=0.1):
        for idx in range(len(self.pops)):
            if random.random()<alpha:
                chices = np.random.choice(len(self.parents), 2, p=self.weight,  replace=False)
                dad, mom = self.parents[chices[0]], self.parents[chices[1]]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                length = min(len(dad[0]), len(mom[0]))
                pick = random.randint(0, len(self.core_cfg)-1)
                for i in range(len(mom[0])):
                    if  mom[0][i] == pick and dad[0][i]==pick:
                        pass
                    elif mom[0][i] == pick:
                        dad[:,i] = mom[:,i]
                    elif dad[0][i] == pick:
                        dad[0][i] = random.randint(0, len(self.core_cfg)-1)
                self.pops[idx] = dad

    def crossover_rg(self, alpha=0.1):
        for idx in range(0,len(self.pops),2):
            if random.random()< alpha:
                chices = np.random.choice(len(self.parents), 2, p=self.weight,  replace=False)
                dad, mom = self.parents[chices[0]], self.parents[chices[1]]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                length = min(len(dad[0]), len(mom[0]))
                point = random.randint(0, length - 1)
                dad[:,:point], mom[:,point:] = mom[:,:point], dad[:,point:]
                self.pops[idx] = dad
                self.pops[idx+1] = mom





    def crossover(self, alpha=0.5):

        for idx in range(0,len(self.pops),2):
            chices = np.random.choice(len(self.parents), 2, p=self.weight, replace=False)
            dad, mom = self.parents[chices[0]], self.parents[chices[1]]
            # dad, mom = self.parents[random.randint(0, len(self.parents) - 1)], self.parents[random.randint(0, len(self.parents) - 1)]
            dad = copy.deepcopy(dad)
            mom = copy.deepcopy(mom)
            if random.random() < alpha:
                length = min(len(dad[0]), len(mom[0]))
                if random.random()<0.5:
                    pick = 0
                else:
                    pick = 1
                point = random.randint(0, length - 1)
                # dad[pick][:point], mom[pick][point:] = mom[pick][:point], dad[pick][point:]
                # dad[pick][:point], mom[pick][:point] = mom[pick][:point], dad[pick][:point]
                if point < length//2:
                    dad[pick][:point], mom[pick][:point] = mom[pick][:point], dad[pick][:point]
                else:
                    dad[pick][point:], mom[pick][point:] = mom[pick][point:], dad[pick][point:]
            self.pops[idx] = dad
            self.pops[idx+1] = mom


    def get_timepoint_set(self, BwTimeTrack):
        timeSet = set()
        for core_jobs in BwTimeTrack:
            for st, end, bw in core_jobs:
                timeSet.add(st)
                timeSet.add(end)
        return timeSet


    def post_process(self, maxBW=1, EPSILON=1e-12, plot=False):
        '''
        job = [left_auc, req_dram_bw, adj_dram_bw]
        '''
        plot_x = []
        plot_h = []
        plot_w = []
        plot_job_id = []
        job_queue = [self.coreJobs[i].pop(0) if len(self.coreJobs[i])>0 else  [float("Inf"),0, -1] for i in range(len(self.coreJobs))]
        reqBW_list = np.array([job[1] for job in job_queue])
        leftAuc_list = np.array([job[0] for job in job_queue])
        jobID_list = np.array([job[2] for job in job_queue])
        curTime = 0
        timeTrack  = np.zeros((len(self.coreJobs)))
        wasteBwArea = 0
        while any(leftAuc_list!=float("Inf")):
            reqBW = np.sum(reqBW_list)
            if reqBW > maxBW:
                adjBW_list = reqBW_list * maxBW / reqBW
            else:
                adjBW_list = reqBW_list

            wasteBW = maxBW - sum(adjBW_list)
            if wasteBW < -1:
                print("error")
            latency_list = leftAuc_list / adjBW_list
            run_duration = np.min(latency_list)
            if plot:
                plot_x.append(curTime)
                plot_h.append(np.cumsum(adjBW_list))
                plot_w.append(run_duration)
                plot_job_id.append(copy.deepcopy(jobID_list))
            wasteBwArea += (wasteBW * run_duration)
            curTime += run_duration
            latency_list -= run_duration
            leftAuc_list -= (run_duration * adjBW_list)
            leftAuc_list[abs(leftAuc_list)<EPSILON] = 0
            next_core_id = np.argmin(latency_list)
            if len(self.coreJobs[next_core_id]) > 0:
                next_job = self.coreJobs[next_core_id].pop(0)
            else:
                next_job = [float("Inf"),0, -1]
                timeTrack[next_core_id] = curTime
            left_auc, req_dram_bw, job_id = next_job
            leftAuc_list[next_core_id] = left_auc
            reqBW_list[next_core_id] = req_dram_bw
            jobID_list[next_core_id] = job_id
        BW_util = 1 - wasteBwArea/(maxBW * curTime)
        if plot:
            return plot_x,  plot_h, plot_w, plot_job_id
        else:
            return curTime, timeTrack, BW_util

    def get_util_rate(self, timeTrack):
        area = np.max(timeTrack) * len(timeTrack)
        util_rate = np.sum(timeTrack)/area
        return util_rate


    def get_core_stat(self, pop):
        coreJobs = [[] for _ in range(len(self.core_cfg))]
        for i in range(len(pop[0])):
            core_id = int(pop[0][i])
            order_index = pop[1][i]
            latency = self.job_table[i][0][core_id]
            req_dram_bw = self.job_table[i][1][core_id]
            coreJobs[core_id].append([order_index, i, latency, req_dram_bw])
        self.coreJobs = self.sorting_corejob(coreJobs)

    def evaluate(self ):
        maxBW = self.dram_bw
        self.BwTimeTrack = [[] for _ in range(len(self.core_cfg))]
        self.coreTimeTrack = np.zeros((len(self.core_cfg),))
        for p, pop in enumerate(self.pops):
            self.get_core_stat(pop)
            maxSpan, timeTrack, BW_util  = self.post_process(maxBW)
            self.latencys[p] = maxSpan
            self.fitness[p] = -maxSpan
            self.utils[p] = self.get_util_rate(timeTrack)
            # print((np.max(timeTrack) - np.min(timeTrack))/np.max(timeTrack))
            self.timeTracks[p] = timeTrack
    def create_job(self):
        jobs = []
        file = self.get_insts_table_file_name()
        if self.is_random and os.path.exists(file) and self.true_random==False:
            self.jobs=jobs
            return
        for lay in range(self.traffic_len):
            CONVtype = random.choice(self.CONVtype_choices)
            dim = single_traffic(CONVtype)
            job = [0, 0, np.array(dim)]
            jobs.append(job)
        self.jobs = jobs
    def get_insts_table_file_name(self):
        if not self.is_random:
            file =self.insts_table_file
        else:
            file = os.path.join(self.random_table_dir, "random{}.plt".format((self.epoch//self.train_per_traffic)))
            # file = os.path.join(self.random_table_dir, "random{}.plt".format((self.epoch//self.train_per_traffic)%1000))
            # file = os.path.join(self.random_table_dir, "random{}.plt".format(self.use_curriculum()))
        return file

    def create_job_table(self, num_jobs_train=False, num_jobs_test=False):
        file = self.get_insts_table_file_name()
        if os.path.exists(file) and self.true_random==False:
            try:
                with open(file, "rb") as fd:
                    temp = pickle.load(fd)
                    self.job_table = temp["tb"] if num_jobs_train is False else temp["tb"][:num_jobs_train,:]
                    if num_jobs_test:
                        self.job_table[num_jobs_test:,:] = float("Inf")
                    return
            except:
                pass
        self.job_table = np.zeros((len(self.jobs), 2, len(self.core_cfg)))
        for j in range(len(self.jobs)):
            for c in range(len(self.core_cfg)):
                _,_, dim = self.jobs[j]
                core_id = c
                self.job_table[j][0][c], self.job_table[j][1][c] = self.get_maestroData(dim, core_id)
        if self.true_random==False:
            with open(file, "wb") as fd:
                temp = {"tb": self.job_table}
                pickle.dump(temp, fd)



    def export_job_table(self,file_name):
        file = self.get_insts_table_file_name()
        jobs = self.jobs
        df = defaultdict(list)
        with open(file, "rb") as fd:
            temp = pickle.load(fd)
            job_table = temp["tb"]
        for i, job_entry in enumerate(job_table):
            for r in range(len(job_entry)):
                for k in range(len(job_entry[0])):
                    df["L-{}_{}".format(i, jobs[i][2])].append(job_table[i][r][k])
        df = pd.DataFrame(df)
        df.to_csv(file_name)
    def get_maestroData(self, dim, action):
        core_id = action
        if self.is_flex:
            latency = self.cost_model.observe_lowestLatency_maestro(dim, core_id, self.is_flex_sel_pe)
        else:
            latency = self.cost_model.observe_maestro_normal(dim, core_id)
        return latency

    def reset(self):
        self.best_rewards = []
        self.best_utils = []
        self.best_cands = []
        self.all_records = []
        self.best_reward = float("-Inf")

    def update_best_reward_list(self, gen, best_cand = None, pops = None, fitness = None):
        cur_reward_id = np.argmax(self.fitness)
        cur_reward = self.fitness[cur_reward_id]
        if gen == 0:
            self.best_reward = cur_reward
        else:
            # self.reward_rec.append(cur_reward)
            if cur_reward > self.best_reward:
                self.best_reward = cur_reward
        self.best_rewards.append(self.best_reward)
        self.best_cands.append(best_cand)
        self.best_utils.append(self.utils[cur_reward_id])
        self.all_records.append([pops, fitness]) if self.save_all_records else None
    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_utils= chkpt["best_utils"]
        self.best_cands= chkpt["best_cands"]
        self.all_records= chkpt["best_caall_recordsnds"]
        self.pops = chkpt["pops"]
    def get_chkpt(self):
        return {
            "reward_rec": self.reward_rec,
            "best_rewards": self.best_rewards,
            "best_utils": self.best_utils,
            "best_reward": self.best_reward,
            "best_cands": self.best_cands,
            "all_records": self.all_records,
            "pops":self.pops
        }
    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)


    def sort_job_by_latency(self):
        lat = np.average(self.job_table[:,0,:], axis=1)
        sort_id = np.argsort(lat)
        return sort_id


    def heu_get_nextcoreid(self, job_id, coreTimeTrack, alg="GREEDY"):

        if alg=="GREEDY":
            return np.argmin(coreTimeTrack)
        elif alg == "RR":
            return job_id%len(self.core_cfg)
        elif alg == "MET":
            min_value = np.min(self.job_table[job_id, 0, :])
            cand = self.job_table[job_id,0,:]
            cand_id = (cand==min_value).nonzero()[0]
            return np.random.choice(cand_id, 1)
        elif alg == "HEFT":
            pred_coreTimeTrach = coreTimeTrack + self.job_table[job_id, 0, :]
            return np.argmin(pred_coreTimeTrach)
        elif alg =="RANDOM":
            return random.randint(0, len(self.core_cfg)-1)
    def heu_get_jobid(self,alg="SJF"):
        distribution = np.arange(0,  1,1 / len(self.job_table))
        if alg == "SJF":
            self.sorting_job()
            return distribution
        if alg =="FCFS":
            return distribution
        if alg =="HEFT":
            sort_id = self.sort_job_by_latency()
            new_distribution = np.zeros((len(self.job_table),))
            for d_id, s_id in enumerate(sort_id):
                new_distribution[s_id] = distribution[d_id]
            return new_distribution
        elif alg=="RANDOM":
            return np.random.random((len(self.job_table),))

    def get_heu_pop(self, alg_jobid, alg_coreid):
        coreTimeTrack = np.zeros((len(self.core_cfg),))
        job_id_distr = self.heu_get_jobid(alg_jobid)
        pop = [[None for _ in range(len(self.job_table))] for _ in range(2)]
        for i in range(len(self.job_table)):
            next_core_id = self.heu_get_nextcoreid(i, coreTimeTrack, alg_coreid)
            latency = self.job_table[i][0][next_core_id]
            coreTimeTrack[next_core_id] += latency
            pop[0][i] = next_core_id
            pop[1][i] = job_id_distr[i]
        self.pops = [pop]
    def run_heursitic(self, alg_jobid = "FCFS", alg_coreid = "GREEDY"):
        self.fitness = np.zeros((1,))
        self.latencys = np.ones((1,))
        self.utils = np.ones((1,))
        self.timeTracks = np.zeros((1, self.num_cores))
        self.create_job_table()
        self.get_heu_pop(alg_jobid, alg_coreid)
        self.evaluate()
        print("[{}-{}] Fitness: {:.2e}".format(alg_jobid, alg_coreid, np.max(self.fitness)))
        return np.max(self.fitness)

    def plot_heurstic(self, alg_jobid = "FCFS", alg_coreid = "GREEDY", image_file=None):
        self.fitness = np.zeros((1,))
        self.create_job_table()
        self.get_heu_pop(alg_jobid, alg_coreid)
        self.evaluate()
        self.plot_pop(self.pops[0], self.fitness[0], image_file=image_file)
