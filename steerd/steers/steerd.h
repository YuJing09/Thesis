#include "common/mat.h"
#include "common/util.h"
#include "common/modeldata.h"
#include "common/visionbuf.h"
#include "common/visionipc.h"
#include "common/timing.h"

#include "commonmodel.h"
#include "runners/run.h"
#include <opencv2/opencv.hpp>
#include <czmq.h>
#include <memory>
#include "messaging.hpp"

#define MODEL_WIDTH 320
#define MODEL_HEIGHT 160
#define MODEL_FRAME_SIZE MODEL_WIDTH * MODEL_HEIGHT * 3/2
#define MODEL_NAME "steering_angle.dlc"

#define MODEL_FREQ 20

struct SteerData {
       float* point;
       float* valid_len;
       float* error;
       float* radar;
};

typedef struct myModelState {
  ModelFrame frame;
  float* output;
  float* input_frame;
  RunModel *m;
} myModelState;
void mymodel_init(myModelState* s,cl_device_id device_id,cl_context context );

SteerData model_eval_frame(myModelState *s,cl_command_queue q,VIPCBuf *buf);
void mymodel_free(myModelState* s);

int myclamp(int value);

cv::Mat getFlatVector(VIPCBuf *buf);

void mymodel_publish(PubMaster &pm, uint32_t vipc_frame_id, uint32_t frame_id,
                   uint32_t vipc_dropped_frames, float frame_drop, const SteerData &net_outputs, uint64_t timestamp_eof);
