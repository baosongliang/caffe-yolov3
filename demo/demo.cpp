
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include "detector.h"
#include <map>
#include <vector>
#include <string>
#include <string.h>
#include <dirent.h>

using namespace cv;


bool signal_recieved = false;


void sig_handler(int signo){
    if( signo == SIGINT ){
            printf("received SIGINT\n");
            signal_recieved = true;
    }
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

typedef std::map<std::string,bool> GroundTureMap;

// prepare file according to label set
void PrepareFile(std::string testImgSet,std::string label_dir, std::string saved_name)
{
    char file_name[512]{0};
    char lable_info[128]{0};

    std::ifstream img_set(testImgSet.c_str(), std::ifstream::in);
    std::ofstream ofs(saved_name.c_str(),std::ofstream::out);
    while (true)
    {
        memset(file_name,0, sizeof(file_name));
        img_set.getline(file_name, sizeof(file_name));
        if (0 == img_set.gcount())
        {
            img_set.close();
            break;
        }

        if (file_name[0] == '\n')
            continue;
        
        LOG(INFO) << "Img:" << file_name ;

        // got label info of that image
        std::string file_str(file_name);
        std::size_t pos_end = file_str.find_last_of(".");
        std::size_t pos_start = file_str.find_last_of("/") + 1;
        std::string label_file_str = label_dir + "/" + file_str.substr(pos_start,pos_end - pos_start) + ".txt";
        LOG(INFO) << "label file:" << label_file_str ;

        // read the target label info to get ground true value.
        int val = 0;
        std::ifstream if_imginfo(label_file_str.c_str(), std::ifstream::in);
        memset(lable_info,0,sizeof(lable_info));
        if_imginfo.getline(lable_info,sizeof(lable_info));

        if (if_imginfo.gcount() > 1)
            val = 1;
        if_imginfo.close();       

        ofs << std::string(file_name) << "=" << val << "\n";
    }

    ofs.close();
}

void LoadGroundTrue(std::string img_info_file,  GroundTureMap & map)
{
    map.clear();

    // get file count
    std::ifstream img_info_fs(img_info_file.c_str(), std::ifstream::in);
    char info_buf[512]{0};
    std::vector<std::string> vals;

    while (true)
    {
        memset(info_buf,0,sizeof(info_buf));
        img_info_fs.getline(info_buf, sizeof(info_buf));
        if(0 == img_info_fs.gcount())
        {
            img_info_fs.close();
            break;
        }

        char *pch = strtok(info_buf, "=");
        vals.clear();
        while (pch != NULL)
        {
            vals.push_back(pch);
            pch = strtok(NULL, "=");
        }

        // ignore the empty line
        if(vals.empty())
            continue;

        std::string img_file = vals[0];
        bool has_person = (bool)(atoi(vals[1].c_str()));

        if (!img_file.empty())
        {
            map[img_file] = has_person;
            LOG(INFO) << "Image:" << img_file << (has_person ? "  has person!" : "  no person!");
        }
    }

}

int TestAccurate(int argc, char** argv)
{
    std::string model_file;
    std::string weights_file;
    std::string image_info_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_info_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_info_path]";
        return -1;
    }

    // PrepareFile(image_info_path,"/root/data/metro_clear_person_merge_20200930/VOCdevkit/VOC2007/labels", "./ground_truth.txt");
    // return 0;

    GroundTureMap ground_true_maps;
    LoadGroundTrue(image_info_path,ground_true_maps);
    if(ground_true_maps.size() < 1)
        return -1;

    LOG(INFO) << "Test total image count:"<< ground_true_maps.size();

    int gpu_id = 0;
    //init network
    Detector detector = Detector(model_file,weights_file,gpu_id);


    // do predict
    float thresh = 0.5;
    int err_cnt1 = 0;   // 'ground true' has person, but predict result is no person
    int err_cnt2 = 0;   // 'ground true' has no person, but predict result has person.

    for ( auto real : ground_true_maps)
    {
        //load image with opencv
        Mat img = imread(real.first);
        std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);
        bool has_person_predict = !bbox_vec.empty();
        if( has_person_predict != real.second )
        {
            LOG(ERROR) << real.first << "  !!!! Ground truth is "<< real.second << " but prdict is " << has_person_predict;

            if( real.second )
                ++err_cnt1;
            else
                ++err_cnt2;
        }

    }
    
    float err_rate =  (err_cnt1 + err_cnt2) * 1.0f / float(ground_true_maps.size());
    LOG(ERROR) << "Accuracy is:"<< (100.0f - err_rate * 100.0f);
    return 0;
}


int TestVideo(int argc, char** argv)
{
    std::string model_file;
    std::string weights_file;
    std::string video_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        video_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [video_path]";
        return -1;
    }

    int gpu_id = 0;
    //init network
    Detector detector = Detector(model_file,weights_file,gpu_id);

    // do predict
    float thresh = 0.5;
    VideoCapture capture;
    Mat frame;
    frame= capture.open(video_path);
    if(!capture.isOpened())
    {
        LOG(ERROR) << "can not open video: "<< video_path;
        return -1;
    }
    string wname = "fish eye";
    namedWindow(wname);
    // font
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0,1.0,0,2,8);

    while (capture.read(frame))
    {
        std::vector<bbox_t> bbox_vec = detector.detect(frame,thresh);
        IplImage tmp = IplImage(frame);
        CvArr *arr = (CvArr *)&tmp;
        //show detection results
        for (int i = 0; i < bbox_vec.size(); ++i)
        {
            bbox_t b = bbox_vec[i];

            int left = b.x;
            int right = b.x + b.w;
            int top = b.y;
            int bot = b.y + b.h;
            rectangle(frame, Point(left, top), Point(right, bot), Scalar(0, 0, 255), 3, 8, 0);
            LOG(INFO) << " label = " << b.obj_id
                      << " prob = " << b.prob
                      << " left = " << left
                      << " right = " << right
                      << " top = " << top
                      << " bot = " << bot;
            std::stringstream info;
            info << b.prob << std::flush;
            cvPutText(arr, info.str().c_str(), Point(left, top), &font, cvScalar(255, 0, 0, 1));
        }

        imshow(wname, frame);
        waitKey(10);
    }
    capture.release();
    return 0;
}

void GetFiles(std::string path, std::vector<std::string> &files)
{
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        return;
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8) ///file
        {
            // std::string strFile;
            // strFile = path;
            // strFile += "/";
            // strFile += ptr->d_name;
            // files.push_back(strFile);

            files.push_back(ptr->d_name);
        }
        else
        {
            continue;
        }
    }
    closedir(dir);
}

int TestImgDir(int argc, char** argv)
{
    std::string model_file;
    std::string weights_file;
    std::string image_dir;
    std::string save_dir;
    if(5 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_dir = argv[3];
        save_dir = argv[4];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_dir] [save_dir]";
        return -1;
    }

    int gpu_id = 0;
    float thresh = 0.2;
    // font
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0,1.0,0,2,8);

    //init network
    Detector detector = Detector(model_file, weights_file, gpu_id);

    std::vector<std::string> imgs;
    GetFiles(image_dir,imgs);

    std::cout << "Total has " << imgs.size() << " images!!" << std::endl;
    for (size_t i = 0; i < imgs.size(); i++)
    {
        string file_path = image_dir + "/" + imgs[i];
        std::cout << " read file:" << file_path << std::endl;

        Mat img = imread(file_path);
        std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);
        IplImage tmp = IplImage(img);
        CvArr *arr = (CvArr *)&tmp;
        //show detection results
        for (int i = 0; i < bbox_vec.size(); ++i)
        {
            bbox_t b = bbox_vec[i];

            int left = b.x;
            int right = b.x + b.w;
            int top = b.y;
            int bot = b.y + b.h;
            rectangle(img, Point(left, top), Point(right, bot), Scalar(0, 0, 255), 3, 8, 0);
            LOG(INFO) << " label = " << b.obj_id
                      << " prob = " << b.prob
                      << " left = " << left
                      << " right = " << right
                      << " top = " << top
                      << " bot = " << bot;
            std::stringstream info;
            info << b.prob << std::flush;
            cvPutText(arr, info.str().c_str(), Point(left, top), &font, cvScalar(255, 0, 0, 1));
        }

        imwrite(save_dir + "/" + imgs[i], img);
    }
}

int SaveLabelInfo( int argc, char** argv)
{
    std::string model_file;
    std::string weights_file;
    std::string image_dir;
    std::string save_dir;
    if(5 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_dir = argv[3];
        save_dir = argv[4];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_dir] [save_dir]";
        return -1;
    }

    int gpu_id = 0;
    float thresh = 0.2;
    // font
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0,1.0,0,2,8);

    //init network
    Detector detector = Detector(model_file, weights_file, gpu_id);

    std::vector<std::string> imgs;
    GetFiles(image_dir,imgs);

    std::cout << "Total has " << imgs.size() << " images!!" << std::endl;
    for (size_t i = 0; i < imgs.size(); i++)
    {
        string info_file = save_dir + "/" + imgs[i].substr(imgs[i].size() - 4);
        std::ofstream ofs(info_file.c_str(),std::ofstream::out);

        string file_path = image_dir + "/" + imgs[i];
        std::cout << " read file:" << file_path << std::endl;

        Mat img = imread(file_path);
        std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);

        std::stringstream info;
        info << file_path << "\n";

        //show detection results
        for (int i = 0; i < bbox_vec.size(); ++i)
        {
            bbox_t b = bbox_vec[i];

            int left = b.x;
            int right = b.x + b.w;
            int top = b.y;
            int bot = b.y + b.h;
            rectangle(img, Point(left, top), Point(right, bot), Scalar(0, 0, 255), 3, 8, 0);

            info << left << "," << top << "," << right << "," << bot << "\n";
        }

        ofs << info.str();
        ofs.close();
    }    
}

int main( int argc, char** argv )
{
    return TestImgDir(argc,argv);
    // return TestVideo(argc,argv);
    // return TestAccurate(argc, argv);

    std::string model_file;
    std::string weights_file;
    std::string image_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_path]";
        return -1;
    }	
    int gpu_id = 0;
    //init network
    Detector detector = Detector(model_file,weights_file,gpu_id);

    //load image with opencv
    Mat img = imread(image_path);
    
    //detect
    float thresh = 0.3;
    std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);


    // font
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0,1.0,0,2,8);

    IplImage tmp=IplImage(img); 
    CvArr* arr = (CvArr*)&tmp;

    //show detection results
    for (int i=0;i<bbox_vec.size();++i){
        bbox_t b = bbox_vec[i];

        int left  = b.x;
        int right = b.x + b.w;
        int top   = b.y;
        int bot   = b.y + b.h;
        rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
        LOG(INFO) << " label = " << b.obj_id
                  << " prob = " << b.prob
                  << " left = " << left
                  << " right = " << right
                  << " top = " << top
                  << " bot = " << bot;
	std::stringstream info;
	info << b.prob << std::flush;
        cvPutText(arr,info.str().c_str(),Point(left,top),&font,cvScalar(255,0,0,1));
	
    }

    ////////show with opencv
   // namedWindow("show",CV_WINDOW_AUTOSIZE);
    //imshow("show",img);
    //waitKey(0);
    imwrite("/root/data/metro_crowd/test_result.jpg",img);

    LOG(INFO) << "done!!!!!!";
    return 0;
}

