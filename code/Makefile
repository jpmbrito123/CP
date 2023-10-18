CC = gcc
SRC = src/
CFLAGS = -O3 -ftree-vectorize -fno-omit-frame-pointer -ffast-math -mavx -march=native 
#-O3 -ftree-vectorize -msse4 -fno-omit-frame-pointer -pg

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.cpp
	$(CC) $(CFLAGS) $(SRC)MD.cpp -lm -o MD.exe

clean:
	rm ./MD.exe

run:
	./MD.exe < inputdata.txt
