#include <PCA.h>


/**
 * @param src ����ͼƬ��������
 * @param eigenVector �����������������
 * @param inputAvgVector ��ֵ��������
 * @param m eigenVector��ά��
 */
int judge(const cv::Mat &src,const cv::Mat eigenVector[], const cv::Mat inputAvgVector[], int m);


/**
 * @param src �������飬ÿ��Ԫ����һ��ͼƬ
 * @param cls src��Ԫ�ظ���
 * @param outputEigenVector �����������������
 * @param outputAvgVector ����ľ�ֵ������������
 * @param threshold PCA����ֵ��Ĭ��0.95
 * @return void, always 0
 */
int train(const cv::Mat src[],size_t cls, cv::Mat outputEigenVector[],cv::Mat outputAvgVector[],double threshold = 0.95);