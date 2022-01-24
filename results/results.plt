set style data lines 
set xrange [0:10] 
set yrange [192.44331351435233: 2235.166842136211] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Results for test-ring-square36.tsp' 
plot 'results.txt' using 1:2 with lines lc 7 lw 2 title 'Best tour',\
'results.txt' using 1:3 with lines lc 3 lw 2 title 'Average tour' 
