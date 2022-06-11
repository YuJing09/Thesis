#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include "common/timing.h"
#include "common/params.h"
#include "steerd.h"
//const int horizontal_crop = 0;
//const int top_crop = 0;
//const int hood_crop = 0;
using namespace cv;
double mt1,mt2,mt3;
int myclamp(int value) {
    return value<0 ? 0 : (value>255 ? 255 : value);
}
Mat getFlatVector(VIPCBuf  *buf) {
    // returns RGB if returnBGR is false
    //int original_shape[3]={874,1164,3};
    //const size_t width = original_shape[1];
    //const size_t height = original_shape[0];

    uint8_t *y = (uint8_t*)buf->addr;
    //uint8_t*u = y + (width * height);
    //uint8_t*v = u + (width / 2) * (height / 2);
    Mat src,rgbsrc,dst,rgb_s;
    
    src=Mat(874*3/2,1164,CV_8UC1,y);
    rgbsrc=Mat(874,1164,CV_8UC3);
    
    cvtColor(src,rgbsrc,COLOR_YUV2RGB_IYUV);
    rgb_s=rgbsrc(Rect(0,0,1164,650));
    //std::cout << rgbslice <<std::endl;
    //src=Mat(874,1164,CV_32FC3,bgrVec);
    Size dsize = Size(320,160);
    resize(rgb_s,dst,dsize,0,0,INTER_LINEAR);
    
    
    
    rgb_s.release();
   // rgbslice.release();
   rgbsrc.release();
    src.release();
    
    return dst;
}
Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE - 1> vander;



  

void poly_fit(float *in_pts, float *in_stds, float *out,float *out_point, int valid_len) {
   // References to inputs
   Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1> > pts(in_pts, valid_len);
   Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1> > std(in_stds, valid_len);
   Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> > p(out, POLYFIT_DEGREE - 1);
 
   float y0 = pts[0];
   pts = pts.array() - y0;
 
   // Build Least Squares equations
   Eigen::Matrix<float, Eigen::Dynamic, POLYFIT_DEGREE - 1> lhs = vander.topRows(valid_len).array().colwise() / std.array();
   Eigen::Matrix<float, Eigen::Dynamic, 1> rhs = pts.array() / std.array();
 
   // Improve numerical stability
   Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
   lhs = lhs * scale.asDiagonal();
 
   // Solve inplace
   p = lhs.colPivHouseholderQr().solve(rhs);
 
   // Apply scale to output
   p = p.transpose() * scale.asDiagonal();
  out[3] = y0;
  for (int i=0;i<50;i++){
    out_point[i]=out[3]+out[2]*i+out[1]*i*i+out[0]*i*i*i;
  //printf("p0:%f,p1:%f,p2:%f,p3:%f\n",out[3],out[2],out[1],out[0]);
}
}
void mymodel_init(myModelState* s,cl_device_id device_id,cl_context context){
  frame_init(&s->frame,MODEL_WIDTH,MODEL_HEIGHT,device_id,context);
  s->input_frame = (float*)calloc(MODEL_FRAME_SIZE*2,sizeof(float));
  const int output_size = 105;
  s->output = (float*)calloc(output_size,sizeof(float));
  
  s->m= new DefaultRunModel("../../models/ypreg.dlc",s->output,output_size,USE_GPU_RUNTIME);
  for(int i = 0; i < 50; i++) {
      for(int j = 0; j < POLYFIT_DEGREE - 1; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1) ;
      }
    }

}


SteerData model_eval_frame(myModelState * s,cl_command_queue q,VIPCBuf *buf){
   
   //float* new_frame_buf = frame_prepare(&s->frame, q, yuv_cl, width, height, transform);
  
   Mat rgb_frame_buf = getFlatVector(buf);

   float *data= (float*)malloc(320*160*3*sizeof(float));
   for (int i=0;i<320*160*3;i++){
     data[i]=float(rgb_frame_buf.data[i]);
    
   }
   
   
  
   
  
   memmove(&s->input_frame[0], data, sizeof(float)*MODEL_FRAME_SIZE*2);
   s->m->execute(s->input_frame,MODEL_FRAME_SIZE*2);
   
   //clEnqueueUnmapMemObject(q,s->frame.net_input,(void*)rgb_frame_buf,0,NULL,NULL);
   SteerData net_outputs;
   rgb_frame_buf.release();
   delete data;
   net_outputs.point = &s->output[0];
   net_outputs.valid_len= &s->output[50];
   net_outputs.error= &s->output[51];
   net_outputs.radar= &s->output[101];
   return net_outputs;
}
void mymodel_free(myModelState* s){
  free(s->output);
  free(s->input_frame);
  frame_free(&s->frame);
  delete s->m;
}
void mymodel_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id,
                   uint32_t vipc_dropped_frames, float frame_drop, const SteerData &net_outputs, uint64_t timestamp_eof) {
  // make msg
  MessageBuilder msg;
  MessageBuilder msgg;
  
  float points[50];
  int  valid_len;
  float error[50];
  float radar_x;
  float radar_y;
  float poly_arr[4];
  float point_arr[50];
  float stds[50];
  bool status=true;
  for (int i =0;i<50;i++){
    points[i] = net_outputs.point[i];
    error[i] = exp(net_outputs.error[i]);
    stds[i]=1.;
  }
  valid_len=fmax(0,fmin(net_outputs.valid_len[0]*10,50));
  radar_x=net_outputs.radar[0]*10.;
  radar_y=net_outputs.radar[1];
  poly_fit(points,stds,poly_arr,point_arr,valid_len);
  auto steerd =msg.initEvent().initSteermodel();
  steerd.setPoints(point_arr);
  steerd.setValidlen(valid_len);
  steerd.setError(error);
  steerd.setRadarx(radar_x);
  steerd.setRadary(radar_y);
  printf("valid_len:%dradar_x:%fradar_y:%f\n",valid_len,radar_x,radar_y);

  auto leadone=msgg.initEvent().initRadarState().initLeadOne();
  leadone.setDRel(radar_x);
  leadone.setYRel(radar_y);
  leadone.setStatus(status);

  
  pm.send("radarState",msgg);
  pm.send("mymodel", msg);
}



