set style data lines 
set xrange [0:10] 
set yrange [180.32820621169958: 2251.7964740160323] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Multirun results for test-ring-square36.tsp' 
plot 'm_results.txt' i 0 using 1:2 with lines lc 0 lw 2 title 'Best tour 0',\
'm_results.txt' i 0 using 1:3 with lines dt 3 lc 0 lw 2 title 'Average tour 0',\
'm_results.txt' i 1 using 1:2 with lines lc 1 lw 2 title 'Best tour 1',\
'm_results.txt' i 1 using 1:3 with lines dt 3 lc 1 lw 2 title 'Average tour 1',\
'm_results.txt' i 2 using 1:2 with lines lc 2 lw 2 title 'Best tour 2',\
'm_results.txt' i 2 using 1:3 with lines dt 3 lc 2 lw 2 title 'Average tour 2',\
