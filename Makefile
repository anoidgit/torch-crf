libtcrf.so : libtcrf.cpp
g++ -o libtcrf.so -Ofast -fopenmp -shared -fPIC libtcrf.cpp

clean : 
	rm -fr libtcrf.so
