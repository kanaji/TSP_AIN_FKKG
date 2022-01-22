set style data lines 
set xrange [0:100] 
set yrange [24488.55587025928: 35970.12536808692] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Multirun results' 
plot 'm_results.txt' i 0 using 1:2 with lines lc 0 lw 2 title 'Best tour 0',\
'm_results.txt' i 0 using 1:3 with lines dt 3 lc 0 lw 2 title 'Average tour 0',\
'm_results.txt' i 1 using 1:2 with lines lc 1 lw 2 title 'Best tour 1',\
'm_results.txt' i 1 using 1:3 with lines dt 3 lc 1 lw 2 title 'Average tour 1',\
'm_results.txt' i 2 using 1:2 with lines lc 2 lw 2 title 'Best tour 2',\
'm_results.txt' i 2 using 1:3 with lines dt 3 lc 2 lw 2 title 'Average tour 2',\
