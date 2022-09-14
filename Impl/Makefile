CXX = clang++
#/usr/local/opt/riscv/bin/clang++ -nostdlib++ \
	-isystem /Library/Developer/CommandLineTools/usr/include/c++/v1 \
	-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include \
	-isystem /Library/Developer/CommandLineTools/usr/include \
	-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks
CXXFLAGS = 
#-nostdinc++
LDFLAGS = 
#-L /usr/local/opt/riscv/lib \
	-Wl,-rpath,/usr/local/opt/riscv/lib \
	-lc++
CPPFLAGS = -std=c++14 -Wall
ROOTFLAGS = `root-config --cflags`
BLASDIR = /usr/local/opt/openblas
BLASFLAGS = -L${BLASDIR} -lblas
SRC = ${wildcard *.cxx}

test: ${SRC:%.cxx=%.o}
	${CXX} ${LDFLAGS} -o test $^ $(BLASFLAGS) $(ROOTFLAGS) ${CPPFLAGS}

%.o: %.cxx
	${CXX} ${CXXFLAGS} ${CPPFLAGS} $(ROOTFLAGS) -c $< 

.phony: clean
clean:
	-rm *.o
