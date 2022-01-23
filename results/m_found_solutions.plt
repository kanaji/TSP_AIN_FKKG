set style data lines 
set xrange [-1:2] 
set yrange [24380.168313092992: 24645.44875200443] 
set xlabel 'Experiments' 
set ylabel 'Tour length' 
set xtics 1 
set title 'Multirun found solutions results for berlin52.tsp' 
set boxwidth 0.6 relative 
set style fill solid 
plot 'm_found_solutions.txt' using 1:2 with boxes lc 7 lw 1 title 'Best tour'
