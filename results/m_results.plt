set style data lines 
set xrange [0:3] 
set yrange [25928.72166565092: 32385.32418429174] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Multirun results for berlin52.tsp' 
plot 'm_results.txt' i 0 using 1:2 with lines lc 0 lw 2 title 'Best tour 0',\
'm_results.txt' i 0 using 1:3 with lines dt 3 lc 0 lw 2 title 'Average tour 0',\
'm_results.txt' i 1 using 1:2 with lines lc 1 lw 2 title 'Best tour 1',\
'm_results.txt' i 1 using 1:3 with lines dt 3 lc 1 lw 2 title 'Average tour 1',\
'm_results.txt' i 2 using 1:2 with lines lc 2 lw 2 title 'Best tour 2',\
'm_results.txt' i 2 using 1:3 with lines dt 3 lc 2 lw 2 title 'Average tour 2',\
