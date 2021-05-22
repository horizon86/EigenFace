#include <PCA.h>


/**
 * @param src 测试图片，列向量
 * @param eigenVector 特征向量，按行组成
 * @param inputAvgVector 均值，列向量
 * @param m eigenVector的维数
 */
int judge(const cv::Mat &src,const cv::Mat eigenVector[], const cv::Mat inputAvgVector[], int m);


/**
 * @param src 矩阵数组，每个元素是一类图片
 * @param cls src的元素个数
 * @param outputEigenVector 输出的特征向量矩阵
 * @param outputAvgVector 输出的均值向量，列向量
 * @param threshold PCA的阈值，默认0.95
 * @return void, always 0
 */
int train(const cv::Mat src[],size_t cls, cv::Mat outputEigenVector[],cv::Mat outputAvgVector[],double threshold = 0.95);