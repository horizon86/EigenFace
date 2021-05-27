#include <PCA.h>

cv::Mat PCAeigenVector(const cv::Mat &src, double threshold)
{
    cv::Mat eigenVector, eigenValue, covMatrix;
    if (src.rows <= src.cols)
    {
        // dbg("src.rows < src.cols\n");
        covMatrix = src * src.t();
        //这里计算出的特征向量就是行向量，特征值只有一列，并且特征值是降序保存的
        cv::eigen(covMatrix, eigenValue, eigenVector);
    }
    else
    {
        // dbg("src.rows > src.cols\n");
        covMatrix = src.t() * src;
        cv::eigen(covMatrix, eigenValue, eigenVector);
        //上面求出的特征向量是行向量，计算时要改成列向量，算完后再改成行向量

        //正则化
        cv::Mat eigenVectorNotNormedRow = (src * eigenVector.t()).t();
        cv::Mat eigenVectorNormedRow(eigenVectorNotNormedRow.rows,eigenVectorNotNormedRow.cols,eigenVectorNotNormedRow.type());
        cv::Mat norms(eigenVectorNotNormedRow.rows,1,eigenVectorNotNormedRow.type());
        for (size_t i = 0; i < eigenVectorNotNormedRow.rows; i++)
        {
            norms.at<double>(i,0) = cv::norm(eigenVectorNotNormedRow.rowRange(i,i+1),cv::NORM_L2);
        }

        eigenVectorNormedRow = eigenVectorNotNormedRow / cv::repeat(norms,1,eigenVectorNotNormedRow.cols);
        eigenVector = eigenVectorNormedRow;
        /* 无正则化
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