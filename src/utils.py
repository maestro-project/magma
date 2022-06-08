import numpy as np
import random
import os





def CONV_traffic():
    dim_range = [[1, 256], [1, 256], [3, 225], [3, 225], [1,3], [1, 3], [1]]
    dim = [random.randint(*dim_range[i]) for i in range(2)]
    r = random.randint(*dim_range[2])
    dim += [r, r]
    r = random.choice(dim_range[4])
    dim += [r, r]
    t = random.choice(dim_range[6])
    dim += [t]
    return dim

def GEMM_traffic():
    dim_range = [[8, 128], [8, 128], [8, 128]]
    dim = [random.randint(*dim_range[i]) for i in range(3)]
    dim += [1,1,1,3]
    return dim


def FNN_traffic():
    dim_range = [[8, 512], [8, 812]]
    dim = [random.randint(*dim_range[i]) for i in range(2)]
    dim += [1,1,1,1,0]
    return dim


def single_traffic(CONVtype):
    if CONVtype =="CONV":
        return CONV_traffic()
    elif CONVtype =="GEMM":
        return GEMM_traffic()
    elif CONVtype == "FFN":
        return FNN_traffic()
    else:
        print("Not supported ConvType")
        return None
def get_CONVtype_choices(convtype="CONV"):
    if convtype == "mix":
        return ["CONV", "GEMM", "FFN"]
    else:
        return ["{}".format(convtype)]



