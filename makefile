red_blue: red_blue.c
	mpicc --std=c99 -o $@ $^ 

clean:
	rm red_blue