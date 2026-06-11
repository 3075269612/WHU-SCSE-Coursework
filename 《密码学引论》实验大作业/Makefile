CC=gcc
CFLAGS=-Wall -Wextra -O2 -Wno-deprecated-declarations
GTKFLAGS=$(shell pkg-config --cflags --libs gtk+-3.0)
SSLFLAGS=-lssl -lcrypto

all: sm4_gui

sm4_gui: main.c
	$(CC) $(CFLAGS) main.c -o sm4_gui $(GTKFLAGS) $(SSLFLAGS)

clean:
	rm -f sm4_gui public.pem private.pem *.enc *.dec

.PHONY: all clean
