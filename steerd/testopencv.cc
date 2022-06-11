#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;
int main(){
  int imgsize=250*500*3;
  float *img=(float*)calloc(imgsize,sizeof(float));
  for (int i =0 ; i<imgsize; i++){
      img[i]=rand() %255;
      }
  Mat image=Mat(250,500,CV_32FC3,img);
  Mat Y;
  Size dsize=Size(320,160);
  resize(image,Y,dsize,0,0,INTER_LINEAR);
  printf("rows:%d,cols:%d",Y.rows,Y.cols);
  return(0);
  }
