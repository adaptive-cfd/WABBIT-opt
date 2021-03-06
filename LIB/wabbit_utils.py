import os
from configparser import ConfigParser
import shutil
import time
###########################

ROOT_DIR = os.path.dirname(os.path.abspath("../Makefile"))

def run_wabbit(params_dict, params_inifile, mpicommand, memory, save_log=True, dat_dir=ROOT_DIR+"/data/"):

    params_key = params_dict["key"]
    params_section = params_dict["section"]
    params_value = params_dict["value"]
    inifiles = params_dict["inifiles"]   # this does not have to be the "main" ini file

    # copy blueprint to new dir
    blueprint = ROOT_DIR+"/LIB/WABBIT"
    dirname = params_key.replace(" ", "") + "_" + params_value.replace(" ", '_').replace(";",'')
    if not os.path.exists(dat_dir+dirname):
        destination = shutil.copytree(blueprint, dat_dir+dirname)

        # change parameter in inifile
        parser = ConfigParser()
        for inifile in inifiles:
            copy = shutil.copyfile(ROOT_DIR+"/inifiles/"+inifile, destination+"/"+inifile)
            parser.read(copy)
            parser.set(params_section, params_key, params_value)
            cfgfile = open(copy, 'w')
            parser.write(cfgfile, space_around_delimiters=False)  # use flag in case case you need to avoid white space.
            cfgfile.close()

        # copy main inifile
        shutil.copyfile(ROOT_DIR+"/inifiles/"+params_inifile, destination+"/"+params_inifile)
    
    else:
	# this is a continue run
        destination = dat_dir+dirname

    # move to new dir
    os.chdir(destination)

    # wabbit command
    c = mpicommand + " " + \
        "wabbit " + params_inifile + " " + memory
    if save_log:
        c += " > "+ dirname+".log &"

    # execute FOM
    success = 1
    print(c)
    time.sleep(3)
    success = os.system("make clean")
    time.sleep(3)
    success = os.system("make")
    time.sleep(8)
    success = os.system(c)
    time.sleep(3)
    os.chdir("../")

    return success



