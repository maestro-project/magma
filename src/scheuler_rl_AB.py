
import numpy as np
import gym
from gym import spaces
import os, datetime
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
from scipy.stats import entropy
import random
from utils import *
import copy
from math import floor
style_dict = {"dla": 0, "shi": 1, "eye": 2}
MAX_styles = len(style_dict)
DIM_MAX = [256, 256, 225, 225, 3, 3, 2]
MAX_pe_cfg = 256

USE_FULL_OBSERVE=True

class SchedulerRL_AB(object):
    def __init__(self, num_cores,  cores_cfg, style_cfg, num_inst,  cost_model, dram_bw = 1, traffic_window = 10,
                 jobs=None,tb_dir="./", chkpt_file="try.plt", is_random=False, traffic_len=100,CONVtype_choices=["CONV"],
                 insts_table_file=None, random_table_dir=None, true_random=False, algA="RL", algB="RL", train_per_traffic=10,
                 save_all_records=False):
        super(SchedulerRL_AB, self).__init__()
        self.traffic_len = traffic_len
        self.is_random = is_random
        self.core_cfg = cores_cfg
        self.style_cfg = np.array([style_dict[s] for s in style_cfg], dtype=float)
        self.num_cores = num_cores
        self.coreTimeTrack = np.array([0 for _ in range(num_cores)])
        self.coreThreadTrack = [[] for _ in range(num_cores)]
        self.threadTimeTrack = np.array([0 for _ in range(num_inst)])
        self.total_jobs = len(jobs) if not is_random else self.traffic_len
        self.jobs = jobs
        self.curTime = 0
        self.dram_bw = dram_bw
        self.len_dim = 6
        self.num_CONVtype = 4
        self.traffic_window = traffic_window

        # self.state_size = len(self.core_cfg) * 2 + len_dim
        # self.action_space = spaces.MultiDiscrete([traffic_window, len(cores_cfg)])
        self.algA = algA
        self.algB = algB
        self.config_algComb()
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.state_size,), dtype=np.float32)
        self.cost_model = cost_model
        self.total_idle_time = 0
        self.curStep = 0
        self.metadata = {'render.modes': []}
        self.state = np.zeros((self.state_size,))
        self.reward_range = (-float('inf'), float('inf'))
        self.epoch = 0
        logdir = os.path.join(tb_dir, "{}".format(datetime.now()))
        self.tboard = SummaryWriter(logdir=logdir)
        self.reward_rec = []
        self.best_reward = []
        self.best_rewards = []
        self.best_cands = []
        self.all_records = []
        self.chkpt_file=chkpt_file
        self.CONVtype_choices=CONVtype_choices

        self.job_table = np.zeros((len(self.jobs), len(self.core_cfg)))
        self.insts_table_file = insts_table_file
        self.random_table_dir = random_table_dir
        self.true_random = true_random
        self.algB_value = 0
        self.algA_value = 0
        self.train_per_traffic = train_per_traffic
        self.create_job_table()
        self.sorting_job()

        self.save_all_records = save_all_records
    def config_algComb(self):
        if self.algA=="RL" and self.algB=="RL":
            self.algComb = "RL-RL"
            self.action_space = spaces.MultiDiscrete([self.traffic_window, len(self.core_cfg)])
            if USE_FULL_OBSERVE:
                self.state_size = len(self.core_cfg) + self.traffic_window * len(self.core_cfg)
            else:
                self.state_size = len(self.core_cfg) + self.traffic_window
        elif self.algA!="RL" and self.algB=="RL":
            self.algComb = "A-RL"
            self.action_space = spaces.Discrete(len(self.core_cfg))
            self.state_size = len(self.core_cfg) + len(self.core_cfg)
        elif self.algA!="RL" and self.algB=="fRL":          #fRL: full overservation RL, use traffic window = len(jobs)
            self.algComb = "A-fRL"
            self.action_space = spaces.MultiDiscrete([len(self.core_cfg) for _ in range(self.traffic_window)])
            # self.action_space = spaces.Box(low=0, high=1, shape=(self.traffic_window,), dtype=np.uint8)
            if USE_FULL_OBSERVE:
                self.state_size = len(self.core_cfg) + self.traffic_window * len(self.core_cfg)
            else:
                self.state_size = len(self.core_cfg) + self.traffic_window
        elif self.algB!="RL" and self.algA=="RL":
            self.algComb = "RL-B"
            self.action_space = spaces.Discrete(self.traffic_window)
            if USE_FULL_OBSERVE:
                self.state_size = len(self.core_cfg) + self.traffic_window * len(self.core_cfg)
            else:
                self.state_size = len(self.core_cfg) + self.traffic_window
        else:
            self.algComb = "A-B"
            self.action_space = spaces.Discrete(self.traffic_window)
            if USE_FULL_OBSERVE:
                self.state_size = len(self.core_cfg) + self.traffic_window * len(self.core_cfg)
            else:
                self.state_size = len(self.core_cfg) + self.traffic_window
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
            # file = os.path.join(self.random_table_dir, "random{}.plt".format((self.epoch//self.train_per_traffic)))
            file = os.path.join(self.random_table_dir, "random{}.plt".format((self.epoch)%3000))
            # file = os.path.join(self.random_table_dir, "random{}.plt".format(self.use_curriculum()))
        return file
    def use_curriculum(self):
        item = self.epoch//self.train_per_traffic
        if item < 20:
            divider= 10
        elif item <200:
            divider = 100
        elif item < 400:
            divider = 200
        elif item < 800:
            divider = 400
        elif item < 1600:
            divider = 800
        else:
            divider = 1000
        ret = item% divider
        return ret

    def create_job_table(self, num_jobs_train=False, num_jobs_test=False):
        file = self.get_insts_table_file_name()
        if os.path.exists(file) and self.true_random == False:
            try:
                with open(file, "rb") as fd:
                    temp = pickle.load(fd)
                    self.job_table = temp["tb"] if num_jobs_train is False else temp["tb"][:num_jobs_train, :]
                    if num_jobs_test:
                        self.job_table[num_jobs_test:, :] = float("Inf")
                    return
            except:
                pass
        self.job_table = np.zeros((len(self.jobs), 2, len(self.core_cfg)))
        for j in range(len(self.jobs)):
            for c in range(len(self.core_cfg)):
                _, _, dim = self.jobs[j]
                core_id = c
                self.job_table[j][0][c], self.job_table[j][1][c] = self.get_maestroData(dim, core_id)
        if self.true_random == False:
            with open(file, "wb") as fd:
                temp = {"tb": self.job_table}
                pickle.dump(temp, fd)

    @property
    def get_state(self):
        return self.state

    def get_cur_reward(self):
        return self.cur_reward
    def norm_dim(self, dim):
        coreTypeOneHot = self.get_onehot(dim[-1])
        dim = dim[:-1]
        dim = np.array([dim[i]/DIM_MAX[i] for i in range(len(dim))], dtype=float)
        # dim = self.norm_value(dim)
        dim = (dim - 0.5) * 2
        return np.concatenate((coreTypeOneHot, dim))

    def norm_cfg(self, core_cfg, maxvalue=256):
        core_cfg = np.array([core_cfg[i]/ maxvalue for i in range(len(core_cfg))], dtype=float)
        core_cfg = self.norm_value(core_cfg)
        return core_cfg
    def get_onehot(self,targets ):
        nb_classes = self.num_CONVtype
        one_hot_targets = np.zeros((nb_classes,))
        one_hot_targets[targets] = 1
        return one_hot_targets

    def init_jobs_pool(self):
        self.next_job = self.traffic_window
        self.jobs_pool = []
        for i in range(self.traffic_window):
            self.jobs_pool.append((i, self.job_table[i,0,:]))

    def get_job_pool_dim(self, masked=False, sel=0):
        dims = []
        if self.algComb=="A-RL":
            if sel < len(self.jobs_pool):
                latency = self.jobs_pool[sel][1]
                dims.extend(latency)
        else:
            for i in range(self.traffic_window):
                if i < len(self.jobs_pool) or(masked and sel==i):
                    if USE_FULL_OBSERVE:
                        latency = self.jobs_pool[i][1]
                    else:
                        latency = self.jobs_pool[i][1][-1:]
                else:
                    if USE_FULL_OBSERVE:
                        latency = np.zeros((len(self.core_cfg),))
                    else:
                        latency = np.zeros((1,))
                    latency[:] = -1
                dims.extend(latency)
        dims = np.array(dims)
        dims = self.norm_value(dims)
        return dims
    def reset(self):
        self.last_util_rate = 0
        self.algB_value = 0
        self.algA_value = 0
        if self.is_random:
            self.create_job()
        self.create_job_table()
        self.sorting_job()
        self.init_jobs_pool()
        dims = self.get_job_pool_dim()
        self.state = np.concatenate(( dims,np.zeros((len(self.core_cfg)))))
        # self.state = np.concatenate((self.norm_dim(dim),np.zeros((len(self.core_cfg)*2,))))
        self.coreTimeTrack = np.array([0 for _ in range(self.num_cores)])
        self.curStep = 0

        self.record_job_sel = []
        self.record_accel_sel = []
        self.best_reward = float("Inf")
        return self.state

    def sorting_job(self):
        sorted_array = sorted(((i, j) for j, i in enumerate(self.job_table)),
                              key=lambda job: ([job[0][0][-i] for i in range(1, len(job[0][0]) - 1, 1)]))
        self.job_table = np.array([ele[0] for ele in sorted_array])
        self.job_table_index = np.array(ele[1] for ele in sorted_array)
    def fill_job_pool(self, layer_id):
        if self.algComb == "A-fRL":
            self.jobs_pool = []
            for _ in range(self.traffic_window):
                if self.next_job < len(self.job_table):
                    self.jobs_pool.append((self.traffic_window + self.curStep,self.job_table[self.next_job,0,:]))
                    self.next_job += 1
        else:
            self.jobs_pool.pop(layer_id)
            if self.next_job < len(self.job_table):
                self.jobs_pool.append((self.traffic_window + self.curStep,self.job_table[self.next_job,0,:]))
                self.next_job += 1
    def get_maestroData(self, dim, action):
        core_id = action
        latency = self.cost_model.oberserve_maestro(dim, core_id)
        return latency


    def norm_value(self, A):
        A = np.array(A)
        if not np.std(A) ==0:
            A = (A - np.mean(A))/np.std(A)
        return A

    def get_util_rate(self):
        denominator = np.max(self.coreTimeTrack) * len(self.coreTimeTrack)
        nominee = np.sum(self.coreTimeTrack)
        util_rate = nominee/denominator
        ret = util_rate - self.last_util_rate
        self.last_util_rate = util_rate
        return  ret

    def intermdeiate_reward_function(self, layer_id, core_id):
        # reward = np.min(self.coreTimeTrack) - np.max(self.coreTimeTrack) #reward v2
        # reward = np.min(self.coreTimeTrack) - self.coreTimeTrack[core_id]  # reward v3
        return  - self.coreTimeTrack[core_id]
        # return self.get_util_rate()
        # if len(self.jobs_pool)==0:
        #     return 0
        # coreTime = np.array([self.coreTimeTrack[core_id] for _ in range(len(self.jobs_pool))])
        # # coreTime = np.array([self.jobs_pool[i][core_id] for i in range(len(self.jobs_pool))])
        # for i in range(len(self.jobs_pool)):
        #     coreTime[i] += self.jobs_pool[i][core_id]
        # layer_penalty = coreTime[layer_id] - np.min(coreTime)
        # layerTime = copy.deepcopy(self.coreTimeTrack)
        # for i in range(len(self.core_cfg)):
        #     layerTime[i] += self.jobs_pool[layer_id][i]
        # core_penalty = layerTime[core_id] - np.min(layerTime)
        # return - (layer_penalty + core_penalty)
    def complete_rest_jobs(self, discount=0.99):
        reward = 0
        for i, (job_id,job) in enumerate(self.jobs_pool):
            core_id = np.argmin(self.coreTimeTrack) ## Greedy
            self.record_accel_sel.append(core_id)
            self.record_job_sel.append(job_id)
            latency = job[core_id]   ##FCFS
            self.coreTimeTrack[core_id] += latency
            reward = reward * discount+self.intermdeiate_reward_function(i, core_id)
        return reward
    def alg_A_access(self):
        if self.algA == "FCFS":
            return self.FCFS()
        elif self.algA == "RANDOM":
            return self.RANDOM(self.jobs_pool)
        elif self.algA == "SJF":
            return self.SJF()
        elif self.algA=="RL":
            pass
        else:
            print("Not supported first stage alg")
        return
    def alg_B_access(self):
        if self.algB == "GREEDY":
            return self.GREEDY()
        elif self.algB == "RR":
            return self.RR()
        elif self.algB == "RANDOM":
            return self.RANDOM(self.core_cfg)
        elif self.algB == "RL" or self.algB=="fRL":
            pass
        else:
            print("Not supported second stage alg")
        return
    def RANDOM(self, A):
        return random.randint(0, len(A)-1) if len(A)> 0 else 0
    def RR(self):
        return self.curStep % len(self.core_cfg)
    def GREEDY(self):
        return np.argmin(self.coreTimeTrack)
    def SJF(self):
        return np.argmin(np.array(self.jobs_pool)[:,-1]) if len(self.jobs_pool)>0 else 0
    def FCFS(self):
        return 0

    def get_core_id(self, action):
        if self.algComb=="RL-RL":
            return action[1]
        elif self.algComb =="A-RL" or self.algComb=="A-fRL":
            return action
        else:
            return self.algB_value
    def get_layer_id(self, action):
        if self.algComb=="RL-RL":
            return action[0]
        elif self.algComb =="RL-B":
            return action
        else:
            return self.algA_value

    def config_next_state(self):
        self.algA_value = self.alg_A_access()
        self.algB_value = self.alg_B_access()
        dims = self.get_job_pool_dim(self.algA_value)
        return dims
    def get_norm_final_reward(self, steps):
        reward = -self.get_finishtime()
        reward/= steps
        return reward
    def determine_method(self):
        for _ in range(len(self.jobs)):
            layer_id = self.alg_A_access()
            core_id = self.alg_B_access()
            latency = self.jobs_pool[layer_id]
            self.fill_job_pool(layer_id)
            self.coreTimeTrack[core_id] += latency[core_id]
            self.state =  np.zeros((self.state_size,))
        print(self.get_finishtime())
        self.update_best_reward_list()
        done = True
        reward = 0
        info = {}
        return self.state, reward, done, info

    def update_latency(self, layer_id, core_id):
        if self.algComb != "A-fRL":
            job_id = self.jobs_pool[layer_id][0]
            latency = self.jobs_pool[layer_id][1]
            self.coreTimeTrack[core_id] += latency[core_id]
            reward = self.intermdeiate_reward_function(layer_id, core_id)
        else:
            for i in range(len(self.jobs_pool)):
                layer = i
                core = core_id[i]
                # core = floor((core_id[i]-0.000001) * len(self.core_cfg))
                latency = self.jobs_pool[layer]
                self.coreTimeTrack[core] += latency[core]
            reward = -np.max(self.coreTimeTrack)
        return reward,job_id
    def step(self, action):
        if self.algComb=="A-B":
            return self.determine_method()
        force_done =False
        info = {}
        layer_id = self.get_layer_id(action)
        core_id = self.get_core_id(action)


        if layer_id >= len(self.jobs_pool):  ##When the job queue is not long enough force done
            force_done = True
            reward = self.complete_rest_jobs()
        else:
            reward,job_id = self.update_latency(layer_id, core_id)
            self.record_accel_sel.append(core_id)
            self.record_job_sel.append(job_id)
            self.fill_job_pool(layer_id)
            dims = self.config_next_state()
            self.state = np.concatenate(( dims, self.norm_value(self.coreTimeTrack)))
            self.curStep += 1

        done = bool(self.curStep == (self.total_jobs) or force_done)
        if done:
            self.update_best_reward_list()
            # print("{:.2e}".format(self.best_reward))
            self.save_chkpt()
            info["best_reward"] = self.best_reward
        if done:
            reward += (-self.cur_reward)  # reward v2, v3
            # reward += (self.get_util_rate())
            pass
        return self.state, reward, done, info


    def get_finishtime(self):
        return np.max(self.coreTimeTrack)


    def get_pop(self):
        acc_sel = np.zeros((1, len(self.jobs)))
        priority = np.zeros((1, len(self.jobs)))
        distribtion = np.arange(0,  1,1 / len(self.jobs))

        for i in range(len(self.record_accel_sel)):
            acc_sel[0, self.record_job_sel[i]] = self.record_accel_sel[i]
            priority[0, self.record_job_sel[i]] = distribtion[i]
        pop = np.concatenate((acc_sel, priority), axis=0)
        return pop
    def get_test_result(self):

        pops = [self.get_pop()]
        fitness = self.test(pops)
        latency = -fitness
        return latency

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
        return np.max(self.fitness)

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

    def post_process(self, maxBW=1, EPSILON=1e-12, plot=False):
        '''
        job = [left_auc, req_dram_bw, adj_dram_bw]
        '''
        plot_x = []
        plot_h = []
        plot_w = []
        plot_job_id = []
        job_queue = [self.coreJobs[i].pop(0) if len(self.coreJobs[i]) > 0 else [float("Inf"), 0, -1] for i in
                     range(len(self.coreJobs))]
        reqBW_list = np.array([job[1] for job in job_queue])
        leftAuc_list = np.array([job[0] for job in job_queue])
        jobID_list = np.array([job[2] for job in job_queue])
        curTime = 0
        timeTrack = np.zeros((len(self.coreJobs)))
        wasteBwArea = 0
        while any(leftAuc_list != float("Inf")):
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
            leftAuc_list[abs(leftAuc_list) < EPSILON] = 0
            next_core_id = np.argmin(latency_list)
            if len(self.coreJobs[next_core_id]) > 0:
                next_job = self.coreJobs[next_core_id].pop(0)
            else:
                next_job = [float("Inf"), 0, -1]
                timeTrack[next_core_id] = curTime
            left_auc, req_dram_bw, job_id = next_job
            leftAuc_list[next_core_id] = left_auc
            reqBW_list[next_core_id] = req_dram_bw
            jobID_list[next_core_id] = job_id
        BW_util = 1 - wasteBwArea / (maxBW * curTime)
        if plot:
            return plot_x, plot_h, plot_w, plot_job_id
        else:
            return curTime, timeTrack, BW_util

    def get_util_rate(self, timeTrack):
        area = np.max(timeTrack) * len(timeTrack)
        util_rate = np.sum(timeTrack) / area
        return util_rate

    def update_best_reward_list(self):

        self.cur_reward = self.get_test_result()
        self.best_reward = min(self.best_reward, self.cur_reward)
        self.epoch += 1
        # self.tboard.add_scalar('cur_reward', abs(self.cur_reward), self.epoch)
        # self.tboard.add_scalar('best_reward', abs(self.best_reward), self.epoch)
        # self.tboard.add_scalars("Reward",{"Best": abs(self.best_reward), "Cur": abs(self.cur_reward) }, self.epoch)
        self.reward_rec.append(abs(self.cur_reward))
        self.best_rewards.append(self.best_reward)

        self.all_records.append([self.get_pop(), self.cur_reward]) if self.save_all_records else None
    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_cands= chkpt["best_cands"]
        self.all_records= chkpt["all_records"]

    def get_chkpt(self):
        return {
            "reward_rec": self.reward_rec,
            "best_rewards": self.best_rewards,
            "best_reward": self.best_reward,
            "best_cands": self.best_cands,
            "all_records":self.all_records,
        }
    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)