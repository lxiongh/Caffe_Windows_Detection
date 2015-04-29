MSVC Project to build a mex file on Windows 64 bits, by mjmarin.
================================================================

Note that you have to modify the path of your Matlab installation. I am using:
C:\Program Files\MATLAB\R2014a\extern\

You should be able to run the demo file included in the original Caffe repository after the compilation:
cd <caffe_folder>\matlab\caffe
im = imread('peppers.png');
[scores, maxlabel] = matcaffe_demo(im,1);