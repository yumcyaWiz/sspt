pt:
	g++ -std=c++17 -fopenmp -O3 pt.cpp

nee:
	g++ -std=c++17 -fopenmp -O3 nee.cpp

fullspec:
	g++ -std=c++17 -fopenmp -O3 fullspectrum.cpp

nee_debug:
	g++ -std=c++17 -fopenmp -g nee.cpp
