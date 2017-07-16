libtcrf.so	:	libtcrf.c
gcc -o libtcrf.so -Ofast -fopenmp -shared -fPIC libtcrf.c

clean	:
	rm -fr libtcrf.so

all	:	libtcrf.so
