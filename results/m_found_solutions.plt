set style data lines 
set xrange [-1:3] 
set yrange [81.32820621169958: 301.11497697889877] 
set xlabel 'Experiments' 
set ylabel 'Tour length' 
set xtics 1 
set title 'Multirun found solutions results for test-ring-square36.tsp' 
set boxwidth 0.6 relative 
set style fill solid 
plot 'm_found_solutions.txt' using 1:2 with boxes lc 7 lw 1 title 'Best tour'
