CC     = gcc
LIBS   = dgemm.c
OBJS   = mxm test
CFLAGS = -O3

all: $(OBJS)

mxm:
	$(CC) -o $@ ../$@.c ../$(LIBS) $(CFLAGS)
	
test:
	$(CC) -o $@ ../$@.c ../$(LIBS) $(CFLAGS)
	  
clean:
	rm $(OBJS)