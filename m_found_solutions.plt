set style data lines 
set xrange [-1:2] 
set yrange [24602.674075725474: 25031.842192672157] 
set xlabel 'Experiments' 
set ylabel 'Tour length' 
set xtics 1 
set title 'Results' 
set boxwidth 0.6 relative 
set style fill solid 
plot 'm_found_solutions.txt' using 1:2 with boxes lc 7 lw 1 title 'Best tour'
