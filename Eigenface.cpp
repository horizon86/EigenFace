#include <Eigenface.h>
using cv::Mat;
int judge(const Mat &src,const Mat* eigenVector, const Mat inputAvgVector[], int m)
{
    int ret = 0;
    double err_min = -1;
    Mat srcFloat,srcFloatNormalized;
    src.convertTo(srcFloat,CV_64FC1,1,0);
    // dbg("\n");
    for (size_t i = 0; i < m; i++)
    {
        const Mat &tmp = eigenVector[i];
        srcFloatNormalized = srcFloat - cv::repeat(inputAvgVector[i],1,srcFloat.cols);
        Mat srcHat = tmp.t() * (tmp * srcFloatNormalized);
        double dis = cv::norm(srcFloatNormalized,srcHat,cv::NORM_L2SQR);
        if(err_min == -1 || dis < err_min)
        {
            // dbg("i=%zd,m=%d,err_min=%lf,dis=%lf\n",i,m,err_min,dis);
            err_min =  dis;
            ret = i;
        }
    }
    return ret;
}

int train(const Mat src[],size_t cls, Mat outputEigenVector[],Mat outputAvgVector[],double threshold)
{
    Mat *p = new Mat[cls];
    Mat *reducedP = new Mat[cls];
    
    for (size_t i = 0; i < cls; i++)
    {
        //转double才能计算特征值
        src[i].convertTo(p[i],CV_64FC1);
        //求各列的均值，结果为列向量
        cv::reduce(p[i],outputAvgVector[i],1,cv::REDUCE_AVG);
        //各列减去均值
        reducedP[i] = p[i] - cv::repeat(outputAvgVector[i],1,src[i].cols);
    }
    
    for (size_t i = 0; i < cls; i++)
        outputEigenVector[i] = PCAeigenVector(reducedP[i],threshold);
    delete[] p;
    delete[] reducedP;
    return 0;
}