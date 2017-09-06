#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>
#include <VIPLFaceCrop.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: command input_image [output_cropped_face]" << std::endl;
		return EXIT_FAILURE;
	}

	std::string input_image = argv[1];
	std::string output_cropped_face = input_image + ".crop.png";
	if (argc > 2)
	{
		output_cropped_face = argv[2];
		if (output_cropped_face.find('.') == output_cropped_face.npos)
		{
			output_cropped_face += ".png";
		}
	}

	VIPLFaceDetector FD("face_preprocess/model/VIPLFaceDetector5.1.0.dat");
	VIPLPointDetector PD("face_preprocess/model/VIPLPointDetector5.0.pts5.dat");

	auto mat = cv::imread(input_image, cv::IMREAD_COLOR);

	if (mat.empty())
	{
		std::cerr << "Can not open image: " << input_image << std::endl;
		return EXIT_FAILURE;
	}

	auto image = vipl_convert(mat);

	auto infos = FD.Detect(image);

	if (infos.empty())
	{
		std::cerr << "Can not detect any faces." << std::endl;
		return EXIT_FAILURE;
	}

	// get the max face
	std::partial_sort(infos.begin(), infos.begin() + 1, infos.end(), [](const VIPLFaceInfo &lhs, const VIPLFaceInfo &rhs)
	{
		return lhs.height * lhs.width > rhs.height * rhs.width;
	});

	auto &info = infos[0];

	VIPLPoint points[5];

	PD.DetectLandmarks(image, info, points);

	VIPLPoint mean_shape[5];
	int mean_shape_size;

	// 获取预设的人脸模型
	VIPL::FaceMeanShape(mean_shape, 5, &mean_shape_size, 0);	// id = 0，表示的是使用人脸识别的crop方法，1 是属性识别

	// 可以调用 VIPL::ResizeMeanShape 对人脸模型缩放，从而直接获取对应的人脸裁剪
	int crop_size = mean_shape_size;	// 赋值为原本大小，就是没有进行缩放
	VIPL::ResizeMeanShape(mean_shape, 5, static_cast<double>(crop_size) / mean_shape_size);
	mean_shape_size = crop_size;

	// 设置 final_size，作用与上一个版本的 SDK 相同，此大小为最终输出的人脸大小
	int final_size = mean_shape_size;
	cv::Mat mat_crop(final_size, final_size, CV_8UC(image.channels));
	VIPLImageData image_crop = vipl_convert(mat_crop);

	// VIPLPoint final_points[5];
	VIPLPoint *final_points = nullptr;
	// 进行人脸裁剪并获取最终的特征点，后三个参数都具有默认值
	// 默认使用 LINEAR 采样，默认不获取最后特征点，默认 final_size 和 mean_shape_size 相同
	bool success = VIPL::FaceCrop(image, image_crop, points, 5, mean_shape, mean_shape_size, VIPL::BY_BICUBIC, final_points, final_size);

	// 保存裁剪的人脸
	cv::imwrite(output_cropped_face, mat_crop);

	return success;
}