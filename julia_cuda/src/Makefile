

CPP=g++
CPPFLAGS=-fopenmp

juliaset_cpu: juliaset.o image.o cli_parser.o kernel_cpu.o
	$(CPP) $(CPPFLAGS) -o juliaset_cpu juliaset.o image.o cli_parser.o kernel_cpu.o

juliaset.o: juliaset.cpp image.hpp complex.hpp
	$(CPP) $(CPPFLAGS) -c juliaset.cpp

image.o: image.cpp image.hpp
	$(CPP) $(CPPFLAGS) -c image.cpp

cli_parser.o: cli_parser.cpp cli_parser.hpp
	$(CPP) $(CPPFLAGS) -c cli_parser.cpp

kernel_cpu.o: kernel_cpu.cpp kernel.hpp
	$(CPP) $(CPPFLAGS) -c kernel_cpu.cpp

clean: 
	rm *.o
