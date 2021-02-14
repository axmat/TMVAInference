CXX = clang++
CPPFLAGS = -std=c++11
ROOTFLAGS = `root-config --cflags`
BLASDIR = /usr/include/openblas
BLASFLAGS = -L${BLASDIR} -lopenblas
SRC = ${wildcard *.cxx}

test: ${SRC:%.cxx=%.o}
	${CXX} -o test $^ $(BLASFLAGS) $(ROOTFLAGS) ${CPPFLAGS}

%.o: %.cxx
	${CXX} ${CPPFLAGS} $(ROOTFLAGS) -c $< 

.phony: clean
clean:
	-rm *.o
