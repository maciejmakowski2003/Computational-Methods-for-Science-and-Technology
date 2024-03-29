CC=gcc
CFLAGS=-Wall -g
LIBS=-lgsl -lgslcblas -lm

all: matrix_multiplication

matrix_multiplication: matrix_multiplication.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

test:
	touch time_test.csv
	./matrix_multiplication

clean:
	rm -f matrix_multiplication
	rm -f time_test.csv