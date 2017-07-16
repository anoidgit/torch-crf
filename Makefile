CC = gcc
CFLAGS = -Ofast -march=native -fopenmp -shared -fPIC

libtcrf.so : libtcrf.c
	$(CC) libtcrf.c -o libtcrf.so $(CFLAGS)

clean :
	rm -fr libtcrf.so

all : libtcrf.so
