# gpgpu_work
本项目中的两个文件是对于img2col算法的cuda实现，运行方式为在文件对应文件夹输入command
nvcc -o xx xx.cu
./xx.exe
如果希望可以对于运行过程进行监视，可以使用command
nsys profile --stats=true ./xx.exe
后续在
Nsight System中打开上述命令的输出文件即可
