#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
log_dir=$SCRIPT_DIR"/../log/"
gnuplot_dir=$SCRIPT_DIR"/"

gnuplot_script="\""

# set terminal size
gnuplot_script=$gnuplot_script"set terminal qt size 1400, 900;"

# get value to plot
if [ $# -ne 0 ]; then
  if [ $1 = "" ]; then
    plot_value="all"
  elif [ $1 = "acc" ]; then
    plot_value="acc"
  elif [ $1 = "vel" ]; then
    plot_value="vel"
  else
  plot_value="all"
  fi
else
  plot_value="all"
fi
echo "[plot_value] "$plot_value

# set multiplot
if [ $plot_value = "all" ]; then
  gnuplot_script=$gnuplot_script"set multiplot layout 2, 3;"
fi

# plot
if [ $plot_value = "all" ]; then
  ## all
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"vel_predict.plot';"
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"acc_predict.plot';"
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"pose_predict.plot';"
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"head_ang_vel_predict.plot';"
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"head_ang_predict.plot';"
elif [ $plot_value == "vel" ]; then
  # vel
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"vel_predict.plot';"
elif [ $plot_value == "acc" ]; then
  # acc
  gnuplot_script=$gnuplot_script"load '"$gnuplot_dir"acc_predict.plot';"
fi

# pause
gnuplot_script=$gnuplot_script"pause -1;"

# unset multiplot
if [ $plot_value = "all" ]; then
  gnuplot_script=$gnuplot_script"unset multiplot;"
fi

# execute gnuplot
gnuplot_script=$gnuplot_script"\""
gnuplot_command='gnuplot -e '$gnuplot_script
eval $gnuplot_command
