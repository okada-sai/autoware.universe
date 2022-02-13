# get log dirrectory
current_dir = system("dirname ".ARG0)."/"
log_dir = current_dir."../log/"

# label
set xlabel "time [s]"
set ylabel "velocity [m/s]"

# plot
p log_dir."predict.log" u 1:2 pt 7 ps 0.5 t "true", \
  log_dir."predict.log" u 1:4 pt 7 ps 0.5 t "predicted"
