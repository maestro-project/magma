def get_setting(index):
    if index==0:
        cls_size = 64
        cores_cfg = [256, 128, 96, 64, 32, 16]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index==1:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index == 2 or index==9 or index==11:          #9 for flexible cluster size
        cls_size = 64
        cores_cfg = [32, 32, 32, 32]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index == 3 or index == 13 or index == 14:
        cls_size = 64
        cores_cfg = [32, 32, 32, 32]
        style_cfg = ["dla", "dla", "dla", "eye"]
    elif index == 4 or index == 15  or index == 16:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla","dla","eye"]
    elif index == 5 or index == 17  or index == 18:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 64, 64, 64, 64]
        style_cfg = ["dla", "dla", "dla", "eye", "dla", "dla", "dla", "eye"]
    elif index == 6 or index == 19  or index == 20:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla", "dla", "eye","dla", "dla", "dla", "dla", "dla", "dla", "dla", "eye"]
    elif index == 7:
        cls_size = 64
        cores_cfg = [32, 64, 128, 32, 64, 128]
        style_cfg = ["dla", "dla", "dla", "eye", "eye", "eye"]
    elif index == 8 or index==10 or index==12: #10 for flexible cluster size
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla","dla","dla"]
    elif index == 99:
        cls_size = 64
        cores_cfg = [64,64,64]
        style_cfg = ["dla","eye", "shi"]

    num_cores = len(cores_cfg)
    # style_cfg = ["dla" for _ in range(num_cores)]
    pe_cfg = [a * cls_size for a in cores_cfg]
    cls_cfg = [cls_size for _ in range(num_cores)]
    return num_cores, cores_cfg, pe_cfg, cls_cfg, style_cfg


def get_setting_v3(index):
    if index==0:
        cls_size = 64
        cores_cfg = [256, 128, 96, 64, 32, 16]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index==1:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index == 2 or index==9 or index==11:          #9 for flexible cluster size
        cls_size = 64
        cores_cfg = [32, 32, 32, 32]
        style_cfg = ["dla" for _ in range(len(cores_cfg))]
    elif index == 3:
        cls_size = 64
        cores_cfg = [32, 32, 32, 32]
        style_cfg = ["dla", "dla", "dla", "shi"]
    elif index == 4:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla","dla","shi"]
    elif index == 5:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 64, 64, 64, 64]
        style_cfg = ["dla", "dla", "dla", "shi", "dla", "dla", "dla", "shi"]
    elif index == 6:
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla", "dla", "shi","dla", "dla", "dla", "dla", "dla", "dla", "dla", "shi"]
    elif index == 7:
        cls_size = 64
        cores_cfg = [32, 64, 128, 32, 64, 128]
        style_cfg = ["dla", "dla", "dla", "shi", "shi", "shi"]
    elif index == 8 or index==10 or index==12: #10 for flexible cluster size
        cls_size = 64
        cores_cfg = [128, 128, 128, 128, 128, 128, 128, 128]
        style_cfg = ["dla", "dla", "dla", "dla", "dla", "dla","dla","dla"]


    num_cores = len(cores_cfg)
    # style_cfg = ["dla" for _ in range(num_cores)]
    pe_cfg = [a * cls_size for a in cores_cfg]
    cls_cfg = [cls_size for _ in range(num_cores)]
    return num_cores, cores_cfg, pe_cfg, cls_cfg, style_cfg

def is_flexible(index=0):
    is_flex = False
    sel_pe = False
    if index >=9:
        is_flex = True
    elif index in [11, 12, 14, 16, 18,20]:
        is_flex = True
        sel_pe = True
    return is_flex, sel_pe