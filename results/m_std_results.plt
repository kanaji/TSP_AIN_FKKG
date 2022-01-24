set style data lines 
set xrange [0:3] 
set yrange [25928.72166565092: 32385.32418429174] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Multirun with std results for berlin52.tsp' 
plot 'm_std_results.txt' using 1:2 with lines lc 7 lw 2 title 'Best tour',\
'm_std_results.txt' using 1:2:3 with yerrorbars lc 7 title 'Std. of best tour',\
'm_std_results.txt' using 1:4 with lines lc 3 lw 2 title 'Average tour',\
'm_std_results.txt' using 1:4:5 with yerrorbars lc 3 title 'Std. of average tour'
