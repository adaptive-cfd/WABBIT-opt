##################################################################
##################################################################
#		MAKEFILE for WABBIT-opt
##################################################################
##################################################################
all: 
	@echo -e "\n\n\n"
	@echo -e  "---------------------------"
	@echo -e  "compiling wabbit for you"
	@echo -e  "---------------------------"
	@echo -e "\n\n\n"
	cd ./LIB/WABBIT/ && $(MAKE) all
	@echo -e "\n\n\n"
	@echo -e  "---------------------------"
	@echo -e  "finished compiling wabbit!" 
	@echo -e  "---------------------------"
	@echo -e "\n\n\n"

##################################################################

#================================================================
# Information about the makefile
#================================================================
info:
	@echo -e "\n\n\n"
	@echo -e  "command \t \t info"
	@echo -e  "-------------------------------------------------------"
	@echo -e "\e[1m make\e[0m \t\t \t generates all fortran binaries"
	@echo -e "\e[1m make update-submodules\e[0m  updates all git submodules"
	@echo -e  "-------------------------------------------------------"
	@echo -e "\n\n\n"


.PHONY: doc test directories
#================================================================
# Update all git submodules
#================================================================
update-submodules:
	git submodule foreach git pull origin master
#================================================================
# Install conda environment and activate it
#================================================================
conda-env:
	conda env create -f LIB/environment.yml
##################################################################
# Remarks:
# 1. the @ forces make to execute the command without printing it
# 	 first
# 2. to execute a make in another directory append the command using
#    semicolon (;)
##################################################################
