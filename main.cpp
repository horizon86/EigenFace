#include <Eigenface.h>
#ifdef _DEBUG
#pragma comment(lib, "./opencv_world452d.lib")
#else
#pragma comment(lib, "./opencv_world452.lib")
#endif
using cv::Mat, cv::imread, cv::imshow, cv::imwrite;
using std::cout, std::cin, std::endl, std::string, std::cerr;

const string keys =
    "{r | 0.9 | trainSet percent}"
    "{i | .\\yalefaces\\| inputPath}"
    "{n | 3 | test case classes}"
    "{t | 0.95 |threshold}";

int check(bool f)
{
    if (!f)
    {
        cerr << "error occur" << endl;
        exit(1);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    dbg("start");

    //args parse
    cv::CommandLineParser parse(argc, argv, keys);
    string pathPrev = parse.get<string>("i");
    double trainSetPercent = parse.get<double>("r");
    double threshold = parse.get<double>("t");
    int n = parse.get<int>("n");

    std::vector<string> picPost = {
        "glasses",
        "happy",
        "leftlight",
        "noglasses",
        "normal",
        "rightlight",
        "sad",
        "sleepy",
        "surprised",
        "wink",
        "centerlight"};

    //
    std::vector<std::vector<string>> trainSet;
    std::vector<string> testSet;
    std::vector<int> labels;
    srand(time(nullptr));

    char buf[255];

    dbg("trainset start");

    for (size_t i = 1; i < n + 1; i++)
    { //��ÿ��������ѡ
        trainSet.push_back(std::vector<string>());
        int trainCnt = trainSetPercent * 11;
        bool trainSelMask[11];
        memset(trainSelMask, 0, sizeof(trainSelMask));
        //ѡ���ض�������ѵ����
        for (size_t j = 0; j < trainCnt;)
        {
            int ran = rand() % picPost.size();
            if (!trainSelMask[ran])
            {
                sprintf(buf, "%ssubject%02zd.%s.bmp", pathPrev.c_str(), i, picPost[ran].c_str());
                trainSet[i - 1].push_back(buf);
                // trainSet.push_back(buf);
                trainSelMask[ran] = true;
                dbg(buf);
                j++;
            }
        }
        //ʣ�µ���Ϊ���Լ�
        for (size_t j = 0; j < 11; j++)
        {
            if (!trainSelMask[j])
            {
                sprintf(buf, "%ssubject%02zd.%s.bmp", pathPrev.c_str(), i, picPost[j].c_str());
                testSet.push_back(buf);
                labels.push_back(i);
                dbg(buf);
            }
        }
    }


    //n��������������k*N^2
    Mat *trainResult = new Mat[n];

    //n������ѵ������ÿ��һ��ͼ��N^2*m
    Mat *src = new Mat[n];

    //��ֵ,N^2*1
    Mat *avgVector = new Mat[n];

    //�м�������src[i]���в�־������
    Mat *colVectorsByClass = new Mat[trainSetPercent * 11];

    int kk = 0;

    dbg("���л���ʼ");

    for (auto &it : trainSet)
    {
        int i = 0;

        for (auto &tpic : it)
        {
            dbg("111");
            //ѭ����ȡ�����ͼ���ڴ���
            Mat tmp = imread(tpic, cv::IMREAD_GRAYSCALE);

            check(tmp.data != nullptr);
            dbg(tpic.c_str());
            //���л�����Ϊ������
            Mat colVector = tmp.reshape(0, tmp.rows * tmp.cols);
            //���л����������������
            colVectorsByClass[i] = colVector;
            i++;
        }
        //ÿһ��ƴ�ӳ�һ������������
        cv::hconcat(colVectorsByClass, i, src[kk]);
        kk++;
    }

    dbg("train start");
    train(src, n, trainResult,avgVector, threshold);
    dbg("train finish");

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < trainResult[i].rows; j++)
        {
            Mat newFaceCol, newFace;
            trainResult[i].row(j).convertTo(newFaceCol, CV_8U);
            newFace = newFaceCol.reshape(0, 243);
            sprintf(buf, "./featureFace/%zd-%zd.jpg", i, j);
            imwrite(buf, newFace);
        }
    }

    /*Mat newFaceCol, newFace;
    trainResult[0].row(0).convertTo(newFaceCol, CV_8U);
    newFace = newFaceCol.reshape(0, 243);
    imshow("newFace", newFace);*/
    // cv::waitKey();

    int predictSum = 0;

    for (size_t i = 0; i < testSet.size(); i++)
    {
        Mat judgeSrc = imread(testSet[i], cv::IMREAD_GRAYSCALE);
        check(judgeSrc.data != nullptr);
        Mat judgeCol = judgeSrc.reshape(0, judgeSrc.cols * judgeSrc.rows);
        int judgeRet = judge(judgeCol, trainResult,avgVector, n);
        // cout << testSet[i] << " is judged as in the same class with " << trainSet[judgeRet][0] << endl;
        printf("predict:\t%d\tGT:\t%d\n",judgeRet + 1,labels[i]);
        if(judgeRet + 1 == labels[i])
        predictSum++;
    }

    printf("predict true :%d / %zd\n",predictSum,testSet.size());
    delete[] colVectorsByClass;
    delete[] trainResult;
    delete[] src;
    return 0;
}