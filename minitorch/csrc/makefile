# クソ適当ビルド
libtensor: cpu.cpp tensor.cpp
	gcc -shared -o ../libtensor.so -fPIC tensor.cpp cpu.cpp 

clean:
	rm -f *.so *o ../*.so ../*.o
