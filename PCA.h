#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#define dbg(...) printf(__VA_ARGS__)
#else
#define dbg(...)
#endif
/**
 * @param src ����ά�ľ�������������ɣ������Ǹ�����
 * @param threshold Ĭ��0.95
 * @return Э��������������������������
 */
cv::Mat PCAeigenVector(const cv::Mat & src, double threshold = 0.95);