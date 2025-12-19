# cとpythonをつなげる例
## 共有ライブラリ作成
```
gcc -shared -o add_floats.so -fPIC add_floats.c
```
-shared:共有ライブラリを生成するオプションです。Linuxでは拡張子 .so（shared object）が使われます。このオプションを指定することで、他のプログラムやライブラリから動的にリンクできる共有ライブラリを作成します

-fPIC："Position Independent Code" の略で、位置に依存しないコードを生成するオプションです。共有ライブラリとして使用されるコードはメモリ上のどこに配置されても正しく動作する必要があるため、このオプションが必要です

## pythonとの連携
exec_c_code.pyに書いた
