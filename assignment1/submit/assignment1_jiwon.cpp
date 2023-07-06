#include<proj_velo2cam/projection_jw.h>

void makeCropBox (PCXYZI& Cloud, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax){
    pcl::CropBox<PXYZI> boxfilter;
    boxfilter.setMin(Eigen::Vector4f(xMin, yMin, zMin, NULL));
    boxfilter.setMax(Eigen::Vector4f(xMax, yMax, zMax, NULL));
    boxfilter.setInputCloud(Cloud.makeShared());
    boxfilter.filter(Cloud);
}

void Clustering(const sensor_msgs::PointCloud2ConstPtr& velo_msg){
    // Clear cam_U and cam_V
    cam_U.clear();
    cam_V.clear();

    //ROI
    PCXYZI::Ptr velo_points(new PCXYZI);
    PCXYZI::Ptr retCloud(new PCXYZI);
    pcl::fromROSMsg(*velo_msg, *velo_points);
    makeCropBox(*velo_points, 0, 35, -7, 7, -1.2, 1);
    //ready for clustering
    pcl::search::KdTree<PXYZI>::Ptr tree (new pcl::search::KdTree<PXYZI>); 
    tree->setInputCloud(velo_points);
    vector<pcl::PointIndices> cluster_indices; 
    // std::cout << velo_points->size() << std::endl;
    // //Euclidean Clustering         
    pcl::EuclideanClusterExtraction<PXYZI> ec;           // clustering with Euclidean method
    ec.setInputCloud(velo_points);   	                 // setting ec with inputCloud
    ec.setClusterTolerance(0.13); 	                     // dist between points ..  cur : 30cm
    ec.setMinClusterSize(10);		                     // minSize the number of point for clustering
    ec.setMaxClusterSize(100000);	                     // minSize the number of point for clustering
    ec.setSearchMethod(tree);				             // searching method : tree 
    ec.extract(cluster_indices); 
    
    //Projection
    proj_velo2cam::clustering_msg clu_msg;
    int cluster_num = 0;
    // Convert velodyne point cloud to cv::Mat
    cv::Mat velo_points_mat(velo_points->size(), 4, CV_64F);
    for (const auto& cluster : cluster_indices) {
        double cluster_mean_dist = 0;
        int cluster_num_points = 0;
        double cluster_u_max = std::numeric_limits<double>::min(); // initialize to lowest possible value
        double cluster_u_min = std::numeric_limits<double>::max(); // initialize to highest possible value
        double cluster_v_max = std::numeric_limits<double>::min(); // initialize to lowest possible value
        double cluster_v_min = std::numeric_limits<double>::max(); // initialize to highest possible value
        for (const auto& index : cluster.indices) {
            PXYZI pt = velo_points->points[index];
            pt.intensity = cluster_num % 10;
            retCloud->push_back(pt);
            velo_points_mat.at<double>(index, 0) = pt.x;
            velo_points_mat.at<double>(index, 1) = pt.y;
            velo_points_mat.at<double>(index, 2) = pt.z;
            velo_points_mat.at<double>(index, 3) = 1;
            cluster_mean_dist += sqrt(pt.x*pt.x + pt.y*pt.y +pt.z*pt.z);

            // Get cam_u value for current point
            cv::Mat cam_UVZ = projection_matrix_right * velo_points_mat.row(index).t();
            double cam_u = cam_UVZ.at<double>(0, 0) / cam_UVZ.at<double>(2, 0);
            double cam_v = cam_UVZ.at<double>(1, 0) / cam_UVZ.at<double>(2, 0);
            if(cam_u > 0 && cam_u <= 1392 && cam_v > 0  && cam_v <= 512){
                cam_U.push_back(cam_u);
                cam_V.push_back(cam_v);
            }
            // Update cluster_u_max and cluster_u_min if necessary
            // Update cluster_v_max and cluster_v_min if necessary
            if (cam_u > cluster_u_max) {
                cluster_u_max = cam_u;
            }
            if (cam_u < cluster_u_min) {
                cluster_u_min = cam_u;
            }
            if (cam_v > cluster_v_max) {
                cluster_v_max = cam_v;
            }
            if (cam_v < cluster_v_min) {
                cluster_v_min = cam_v;
            }
            cluster_num_points++;
        }
        if (cluster_num_points > 0) {
            cluster_mean_dist /= (float)cluster_num_points;
            // cout << cluster_mean_dist <<endl;
            clu_msg.cluster_ID.push_back(cluster_num);
            clu_msg.cluster_uMax.push_back(cluster_u_max);
            clu_msg.cluster_uMin.push_back(cluster_u_min);
            clu_msg.cluster_vMax.push_back(cluster_v_max);
            clu_msg.cluster_vMin.push_back(cluster_v_min);
            clu_msg.cluster_meanDistance.push_back(cluster_mean_dist);
            cluster_num++;
        }
        
    }
    sensor_msgs::PointCloud2 output; 
    pub_process(retCloud,output); 
    pub_clusteringData.publish(output); 
    pub_cluster_meanDistance.publish(clu_msg);
}

void view_projection(const sensor_msgs::ImageConstPtr& cam_msg){
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cam_msg, sensor_msgs::image_encodings::BGR8); // MONO8 색상 채널 사용 //BGR8: color
    // Draw points on the image
    for (int i = 0; i < cam_U.size(); i++){
        cv::circle(cv_ptr->image, cv::Point(cam_U[i], cam_V[i]), 1, cv::Scalar(0,0,255), -1);
    }
    // Convert cv::Mat to sensor_msgs::Image
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg(); // MONO8 색상 채널 사용
    // Publish the image
    pub_image.publish(msg);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "clustering"); //node name 
	ros::NodeHandle nh;   
    ros::Subscriber sub_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 10, Clustering);
    ros::Subscriber sub_image= nh.subscribe("/kitti/camera_color_right/image_raw",10, view_projection);
    pub_cluster_meanDistance = nh.advertise<proj_velo2cam::clustering_msg>("/cluster_mean_distance",10);
    pub_image = nh.advertise<sensor_msgs::Image>("/projection_img", 10);
    pub_clusteringData = nh.advertise<sensor_msgs::PointCloud2>("/clustering_data",10);
    ros::spin();
}