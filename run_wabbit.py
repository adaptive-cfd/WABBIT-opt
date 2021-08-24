#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 22:20:38 2021

@author: miriam
"""

import os

# Parameter müssen eingelesen werden (werden durch genetischen Algorithmus vorgegeben)
def run_wabbit(params):
    
    # ini file ändern    
    inifile = open("wingsection_inifile1.ini", "w")
    inifile.write("[Wingsection]\ntype=Fourier;\nnfft_x0=1;\nnfft_y0=1;\nnfft_alpha=1;\n")
    inifile.write("a0_x0=" + repr(params[0]) + ";\n")
    inifile.write("ai_x0=" + repr(params[1]) + ";\n")
    inifile.write("bi_x0=" + repr(params[2]) + ";\n")
    inifile.write("a0_y0=" + repr(params[3]) + ";\n")
    inifile.write("ai_y0=" + repr(params[4]) + ";\n")
    inifile.write("bi_y0=" + repr(params[5]) + ";\n")
    inifile.write("a0_alpha=" + repr(params[6]) + ";\n")
    inifile.write("ai_alpha=" + repr(params[7]) + ";\n")
    inifile.write("bi_alpha=" + repr(params[8]) + ";\n")
    inifile.write("section_thickness=0.05;")
        
    # wabbit aufrufen
    c = "mpirun --use-hwthread-cpus wabbit PARAMS_2wingsections.ini --memory=8.0GB"
    print("\n",c,"\n\n")
    success = os.system(c)   # execute command
    if success != 0:
        print("command did not execute successfully")
        
    # Daten auswerten im genetischen Algorithmus
    return
