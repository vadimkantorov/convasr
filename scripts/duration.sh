awk -F ',' '{sum += $3} END {print sum / 3600; print "hours"}' $1
