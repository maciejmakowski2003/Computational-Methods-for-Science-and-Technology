CC=gcc
CFLAGS=-Wall -g
LIBS=-lgsl -lgslcblas -lm

all: computer_arithmetic

computer_arithmetic: computer_arithmetic.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f computer_arithmetic