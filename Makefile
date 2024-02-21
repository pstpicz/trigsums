FILESG	= gmain.c trgsum.c
PROG_G	= goer
FILESR	= rmain.c trgsum.c
PROG_R	= rein
LDLIBS	= 
OPTFLG	= -O3 -qopenmp -march=native -mtune=native -qopt-zmm-usage=high
#-xCORE-AVX512 -qopt-zmm-usage=high -D_XDEBUG
all: 
	icx $(OPTFLG) -o $(PROG_G) $(FILESG) $(LDLIBS)
	icx $(OPTFLG) -o $(PROG_R) $(FILESR) $(LDLIBS)
clean:
	rm -f core *.o

