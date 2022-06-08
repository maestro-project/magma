
import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
import pandas as pd
from math import  ceil

MAX_L2_SIZE = 2*1024*1024

m_type_dicts = {0:"CONV", 1:"CONV", 2:"DSCONV", 3:"CONV"}
CONVtype_dicts = {0:"FFN", 1:"CONV",2:"DSCONV", 3:"GEMM"}
class MaestroEnvironment(object):
    def __init__(self, num_cores = 4, pe_cfg = [64*64] * 4, cls_cfg = [64] * 4, style_cfg = ["dla"]* 4, fitness=["latency", "power", "energy"]):
        super(MaestroEnvironment,self).__init__()
        maestro = "../cost_model/maestro"
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)
        self._executable = "{}{}".format(maestro, random_file_name)
        cmd = "cp {} {}".format(maestro, self._executable)
        os.system(cmd)
        self.num_cores = num_cores
        self.fitness = fitness
        self.pe_cfg = pe_cfg
        self.cls_cfg = cls_cfg
        self.style_cfg = style_cfg
        self.cores = {}
        self.make_cores()
    def make_cores(self):
        for i in range(self.num_cores):
            self.cores[i] = {
                "PE": self.pe_cfg[i],
                "cls": self.cls_cfg[i],
                "style": self.style_cfg[i]
            }



    def get_CONVtypeShape(self, dimensions, CONVtype=1):
        CONVtype = CONVtype_dicts[CONVtype]
        if CONVtype == "CONV"or CONVtype=="DSCONV":
            pass
        elif CONVtype == "GEMM":
            SzM, SzN, SzK,*a = dimensions
            dimensions = [SzN, SzK, SzM, 1, 1, 1]
        elif CONVtype == "FFN":
            SzOut, SzIn, *a = dimensions
            dimensions = [SzOut, SzIn, 1, 1, 1, 1]
        else:
            print("Not supported layer.")
        ret = "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*dimensions)
        return ret
    def write_maestro(self, dimensions, core_id, layer_id=0, m_file=None, cls_size=None):

        m_file = self.random_file_name if m_file is None else m_file
        m_type = m_type_dicts[int(dimensions[-1])]
        core = self.cores[core_id]
        cls = core["cls"] if cls_size is None else cls_size
        layershape = self.get_CONVtypeShape(dimensions, int(dimensions[-1]))
        with open("../m_files/{}_f.m".format(core["style"])) as fd:
            with open("../m_files/dpt_f.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant ClusterSz {};\n".format(cls))
                    fo.write("Constant ClusterSzShi {};\n".format(cls+dimensions[4]-1))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format(m_type))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(layershape)
                    if m_type == "CONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")


    def find_best_alignment(self, dim, core_id):
        style = self.cores[core_id]["style"]
        pe = self.cores[core_id]["PE"]
        K, C, Y, X, R, S, T = dim
        sel_cls_size_list = []
        if style=="dla":
            if CONVtype_dicts[T]=="GEMM":
                K, C = C, Y
            for cls in range(1, min(C+1, pe)):
                if C % cls == 0:
                    l2_cls = pe//cls
                    l2_iter = ceil(K/l2_cls)
                    l2_cls =  ceil(K/l2_iter)
                    aloc_pe = pe if CONVtype_dicts[T]=="DSCONV" else min(pe, l2_cls*cls)
                    sel_cls_size_list.append([cls, aloc_pe])
        elif style=="eye":
            # for cls in range(2, min(Y -R+2, pe)):
            #     sel_cls_size_list.append(cls)
            pass
        elif style == "shi":
            for cls in range(2, min(X - S + 2, pe)):
                sel_cls_size_list.append(cls)
            pass
        else:
            print("Not supported best alignment for [{}] mapping style".format(style))
        return sel_cls_size_list



    def observe_lowestLatency_maestro(self, dimensions, core_id, sel_pe=False):
        sel_cls_size_list = self.find_best_alignment(dimensions, core_id)
        reward_list = []
        latency, req_dram_bw, l2_size =self.oberserve_maestro(dimensions, core_id)
        reward_list.append([latency, req_dram_bw, len(sel_cls_size_list)])
        if l2_size > MAX_L2_SIZE:
            print("Need to change setting")
        # reward_list.append([latency, req_dram_bw])
        for id, (l1_cls, aloc_pe) in enumerate(sel_cls_size_list):
            latency, req_dram_bw, l2_size = self.oberserve_maestro(dimensions, core_id, cls_size=l1_cls, num_pe=aloc_pe if sel_pe else None)
            if l2_size > MAX_L2_SIZE:
                latency = float("Inf")
            reward_list.append([latency, req_dram_bw, id])

        sorted_array = sorted((tuple(rew) for rew in reward_list))
        self.job_table = np.array([ele[0] for ele in sorted_array])
        self.job_table_index = np.array(ele[1] for ele in sorted_array)
        return list(sorted_array[0][:2])

    def observe_maestro_normal(self, dimensions, core_id):
        ret = self.oberserve_maestro(dimensions, core_id)
        return ret[:2]
    def oberserve_maestro(self, dimensions, core_id, cls_size=None, num_pe=None):
        m_file = self.random_file_name
        self.write_maestro(dimensions, core_id, cls_size=cls_size)
        core = self.cores[core_id]
        pe = core["PE"] if num_pe is None else num_pe
        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None

        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw_cstr=81920000",
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--offchip_bw_cstr=81920000",
                   "--noc_mc_support=true", "--num_pes={}".format(int(pe)),
                   "--num_simd_lanes=1", "--l1_size_cstr=819200000",
                   "--l2_size_cstr=819200000", "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]


        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()

        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            req_dram_bw = l2_size/2/(runtime-1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, req_dram_bw]]
            # penalty = 1 / num_pe + 1 / bw
            # penalty= 1/penalty

            if len(str(stderr))>3:
                return None
            return self.judge()
        except:
            return None

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power, req_dram_bw = self.observation
        ret = []
        single_fitness=False
        if type(self.fitness) is not list:
            single_fitness=True
            fitness = [self.fitness]
        else:
            fitness = self.fitness
        for term in fitness:
            if term == "energy":
                reward = energy
            elif term == "req_dram_bw":
                reward = req_dram_bw
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = energy * runtime
            elif term == "LAP":
                reward = area * runtime
            elif term == "EAP":
                reward = area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = runtime
            elif term == "area":
                reward = area
            elif term == "l1_size":
                reward =  l1_size
            elif term == "l2_size":
                reward = l2_size
            elif term == "power":
                reward = power
            else:
                raise NameError('Undefined fitness type')
            ret.append(reward)
        ret = ret if not single_fitness else ret[0]
        return ret
