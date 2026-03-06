CC     = gcc
CFLAGS = -Wall -pedantic -Wextra -std=c99 -msse2 -O3 \
         -ftree-vectorize -ffast-math -funroll-loops -fomit-frame-pointer
# -m64?
LIBS   = -lgsl -lgslcblas -lm -lpthread

tvb_cbv_stim: tvbii_multicore.c
        $(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
        rm -f tvb_cbv_stim