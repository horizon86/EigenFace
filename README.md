[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg?style=flat-square)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
# EigenFace
一个Eigenface算法的实现

使用`opencv 4.5.2`，`msvc 19.28.29912`编译

# 编译命令

## Release版
进入`msvc x64`控制台，将`opencv_world452.lib`放在当前目录，执行
```shell
cl /EHsc /nologo /std:c++17 /O1 /MD /GF /I . /Fe: main.exe main.cpp Eigenface.cpp PCA.cpp
```

## Debug版
进入`msvc x64`控制台，将`opencv_world452.lib`放在当前目录，执行
```shell
cl /EHsc /nologo /std:c++17 /Od /MDd /D _DEBUG /GF /I . /Fe: main.exe main.cpp Eigenface.cpp PCA.cpp
```
# 运行

默认用3个类做测试
```shell
main.exe 
```

使用10个类做测试
```shell
main.exe -n=10
```
指定70%的数据做训练集
```shell
main.exe -r=0.7 -n=10
```
