@echo off
set REPO=%~dp0

call D:\Conda\Scripts\activate.bat D:\Conda\envs\env_isaaclab

set ISAACSIM_PATH=C:\isaac-sim
set ISAACSIM_PYTHON_EXE=%ISAACSIM_PATH%\python.bat
set PYTHONEXE=%CONDA_PREFIX%\python.exe

cd /d D:\IsaacLab
D:\IsaacLab\_isaac_sim\python.bat D:\Project\RLCPG\quadraped_RL_CPG\script\CPGv2.py
