

all: hello sort wavelet

hello:
	gcc hello_world_cl.c CLutilities.c -lOpenCL -I/usr/local/cuda/include -o hello_world_cl

sort:
	gcc sort.c CLutilities.c -lOpenCL -lGL -lglut -I/usr/local/cuda/include -o sort

wavelet:
	gcc cl3.c readppm.c CLutilities.c -lOpenCL -lGL -lglut -I/usr/local/cuda/include -o wavelet

clean:
	rm hello_world_cl
	rm sort
	rm wavelet