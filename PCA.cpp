#include <PCA.h>

cv::Mat PCAeigenVector(const cv::Mat &src, double threshold)
{
    cv::Mat eigenVector, eigenValue, covMatrix;
    if (src.rows <= src.cols)
    {
        // dbg("src.rows < src.cols\n");
        covMatrix = src * src.t();
        //����������������������������������ֵֻ��һ�У���������ֵ�ǽ��򱣴��
        cv::eigen(covMatrix, eigenValue, eigenVector);
    }
    else
    {
        // dbg("src.rows > src.cols\n");
        covMatrix = src.t() * src;
        cv::eigen(covMatrix, eigenValue, eigenVector);
        //�������������������������������ʱҪ�ĳ���������������ٸĳ�������

        //����
        cv::Mat eigenVectorNotNormedRow = (src * eigenVector.t()).t();
        cv::Mat eigenVectorNormedRow(eigenVectorNotNormedRow.rows,eigenVectorNotNormedRow.cols,eigenVectorNotNormedRow.type());
        cv::Mat norms(eigenVectorNotNormedRow.rows,1,eigenVectorNotNormedRow.type());
        for (size_t i = 0; i < eigenVectorNotNormedRow.rows; i++)
        {
            norms.at<double>(i,0) = cv::norm(eigenVectorNotNormedRow.rowRange(i,i+1),cv::NORM_L2);
        }

        eigenVectorNormedRow = eigenVectorNotNormedRow / cv::repeat(norms,1,eigenVectorNotNormedRow.cols);
        eigenVector = eigenVectorNormedRow;
        /* ������
        cv::Mat tttmp = (src * eigenVector.t()).t();
        eigenVector = tttmp;
        */
        // eigenVector = (src * eigenVector.t()).t();
    }
    // dbg("eigenVector has cal.\n");
    double sum = 0, nsum = 0;
    int k = 1;
    for (size_t i = 0; i < eigenValue.rows; i++)
        sum += eigenValue.at<double>(i, 0);
    for (; k <= eigenValue.rows; k++)
    {
        
        nsum += eigenValue.at<double>(k-1, 0);
        if (nsum / sum >= threshold)
            break;
    }
    
    cv::Mat retEigenVector = eigenVector.rowRange(0,k);
    return retEigenVector;
}