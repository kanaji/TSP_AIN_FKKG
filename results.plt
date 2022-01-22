set style data lines 
set xrange [0:100] 
set yrange [25771.64615566643: 35020.117230648546] 
set xlabel 'Generations' 
set ylabel 'Tour length' 
set title 'Results' 
plot 'results.txt' using 1:2 with lines lc 7 lw 2 title 'Best tour',\
'results.txt' using 1:3 with lines lc 3 lw 2 title 'Average tour' 
