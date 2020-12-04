CXX = clang++
CPPFLAGS = -std=c++11
ROOTFLAGS = `root-config --cflags`
BLASDIR = /usr/local/opt/openblas
BLASFLAG = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}

test: ${SRC:%.cxx=%.o}
	${CXX} -o test $^ $(BLASFLAG) $(ROOTFLAGS) ${CPPFLAGS}

%.o: %.cxx
	${CXX} ${CPPFLAGS} $(ROOTFLAGS) -c $< 

.phony: clean
clean:
	-rm *.o
