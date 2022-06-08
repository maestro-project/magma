import os, sys
commit_id = 'e1d8efd8e5469cf865a9db60007a70e3f0cb8778'
dst_path = "cost_model/maestro"
maestro_dir = "../temp/maestro"
working_path = os.getcwd()
dst_path = os.path.join(working_path, dst_path)
maestro = os.path.join(maestro_dir, "maestro")
maestro =  os.path.abspath(maestro)
if os.path.exists(maestro_dir) is False:
    os.system("git clone https://github.com/maestro-project/maestro.git {}".format(maestro_dir))
    os.chdir(maestro_dir)
    os.system(f"git checkout {commit_id}")
    try:
        os.system("scons")
    except:
        "Something wring when building maestro, please check maestro repository installation step"
if os.path.exists(maestro) is False:
    os.chdir(maestro_dir)
    try:
        os.system("scons")
    except:
        "Something wring when building maestro, please check maestro repository installation step"
os.chdir(working_path)
if os.path.exists(dst_path) is False:
    os.symlink(maestro, dst_path)
