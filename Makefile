CXX = clang++
CPPFLAGS = -std=c++14
ROOTFLAGS = `root-config --cflags`
BLASDIR = /usr/local/lib/BLAS-3.8.0
BLASFLAGS = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}

test: ${SRC:%.cxx=%.o}
	${CXX} -o test $^ $(BLASFLAGS) $(ROOTFLAGS) ${CPPFLAGS}

%.o: %.cxx
	${CXX} ${CPPFLAGS} $(ROOTFLAGS) -c $< 

.phony: clean
clean:
	-rm *.o
