#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#define dbg(...) printf(__VA_ARGS__)
#else
#define dbg(...)
#endif
/**
 * @param src 待降维的矩阵，由列向量组成，必须是浮点数
 * @param threshold 默认0.95
 * @return 协方差矩阵的特征向量矩阵，行向量
 */
cv::Mat PCAeigenVector(const cv::Mat & src, double threshold = 0.95);