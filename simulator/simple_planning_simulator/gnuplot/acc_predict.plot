# get log dirrectory
current_dir = system("dirname ".ARG0)."/"
log_dir = current_dir."../log/"

# label
set xlabel "time [s]"
set ylabel "acceleration [m/s^2]"

# plot
p log_dir."predict.log" u 1:3 pt 7 ps 0.5 t "true", \
  log_dir."predict.log" u 1:5 pt 7 ps 0.5 t "predicted"
