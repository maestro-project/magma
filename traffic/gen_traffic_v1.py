import numpy as np
import random
import os





def gen_CONV_traffic(batchdir = "try/", num_layers = 100):
    os.makedirs(batchdir, exist_ok=True)
    dim_range = [[1, 256], [1, 256], [3, 225], [3, 225], [1,3], [1, 3], [1]]
    for tr in range(10):
        outfile_t =  batchdir+"traffic{}_m.csv".format(tr)
        with open(outfile_t, "w") as fd:
            for lay in range(num_layers):
                dim = [random.randint(*dim_range[i]) for i in range(2)]
                r = random.randint(*dim_range[2])
                dim += [r, r]
                r = random.choice(dim_range[4])
                dim += [r, r]
                t = random.choice(dim_range[6])
                dim += [t]
                fd.write("{},{},{},{},{},{},{}\n".format(*dim))

def gen_GEMM_traffic(batchdir = "try/", num_layers = 100):
    os.makedirs(batchdir, exist_ok=True)
    dim_range = [[8, 128], [8, 128], [8, 128]]
    for tr in range(10):
        outfile_t =  batchdir+"traffic{}_m.csv".format(tr)
        with open(outfile_t, "w") as fd:
            for lay in range(num_layers):
                dim = [random.randint(*dim_range[i]) for i in range(3)]
                dim += [1,1,1,3]
                fd.write("{},{},{},{},{},{},{}\n".format(*dim))


def gen_FNN_traffic(batchdir = "try/", num_layers = 100):
    os.makedirs(batchdir, exist_ok=True)
    dim_range = [[8, 512], [8, 812]]
    for tr in range(10):
        outfile_t =  batchdir+"traffic{}_m.csv".format(tr)
        with open(outfile_t, "w") as fd:
            for lay in range(num_layers):
                dim = [random.randint(*dim_range[i]) for i in range(2)]
                dim += [1,1,1,1,0]
                fd.write("{},{},{},{},{},{},{}\n".format(*dim))

gen_FNN_traffic("./batch3/FNN/")
gen_GEMM_traffic("./batch3/GEMM/")
gen_CONV_traffic("./batch3/CONV/")