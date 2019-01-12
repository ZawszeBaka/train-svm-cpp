#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

// DEF
Mat cvtVecOfVec2Mat(vector<vector<float>> angles)
{
    cv::Mat matAngles(angles.size(), angles.at(0).size(), CV_32F);
    for(int i=0; i<matAngles.rows; ++i)
        for(int j=0; j<matAngles.cols; ++j)
            matAngles.at<float>(i, j) = angles.at(i).at(j);
    return matAngles;
}

int main(int argc, char *argv[])
{
    Ptr<SVM> svm ;
    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // INIT
    Ptr<cv::HOGDescriptor> hog;
    vector<string> desc_file;
    vector<int> desc_label;

    /*
      Size winSize
      Size blockSize
      Size blockStride
      Size cellSize
      int nbins
    */
    Size winSize(20,20);
    Size blockSize(4,4);
    Size blockStride(2,2);
    Size cellSize(2,2);
    int nbins = 9;
    hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

    desc_file.push_back("/home/non/Documents/data/left-sign.txt");
    desc_file.push_back("/home/non/Documents/data/right-sign.txt");
    desc_file.push_back("/home/non/Documents/data/non-sign.txt");

    desc_label.push_back(0);
    desc_label.push_back(1);
    desc_label.push_back(2);

    // get train data from 3 description files !
    vector<vector<float>> data;
    vector<int> labels;

    // read each type : left , right, non
    for(int i = 0; i < desc_file.size(); i++)
    {
        ifstream f(desc_file[i]);
        string str;
        while(getline(f,str))
        {
          str = "/home/non/Documents/data/" + str;

          // cout <<"[DEBUG] file name : " << str << "\n";

          Mat rimg = imread(str, IMREAD_GRAYSCALE);
          Mat img;
          resize(rimg,img, winSize);

          // imshow("test",img);
          // waitKey();

          // extracting HOG
          vector<float> desc;
          vector<Point> locations;
          hog->compute(img,desc,Size(0,0),Size(0,0),locations);

          // cout << "[INFO] Amount of HOG descriptor : " << desc.size() << "\n";

          data.push_back(desc);
          labels.push_back(desc_label[i]);
        }
        f.close();
    }

    // convert vector of vector to Mat
    Mat X = cvtVecOfVec2Mat(data);
    Mat y(labels.size(),1,CV_32SC1,labels.data());

    cout << "[INFO] X size: " << X.size().height << " , " << X.size().width << "\n";
    cout << "[INFO] y size: " << y.size().height << " , " << y.size().width << "\n" << "\n";

    cout << "[INFO] Training ... ";
    svm->train(X,ROW_SAMPLE,y);
    cout << "Done!\n";

    svm->save("/home/non/Documents/data/train_svm/model/out.xml");

    return 1;
}
