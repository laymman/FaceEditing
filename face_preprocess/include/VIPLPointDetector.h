#ifndef VIPL_POINT_DETECTOR_H_
#define VIPL_POINT_DETECTOR_H_

#define _SDK_MIN_MSC_VER 1800
#define _SDK_MAX_MSC_VER 1900
#if defined(_MSC_VER)
#if _MSC_VER < _SDK_MIN_MSC_VER || _MSC_VER > _SDK_MAX_MSC_VER
#error "Unsupported MSVC. Please use VS2013(v120) or compatible VS2015(v140)."
#endif // _MSC_VER < 1800 || _MSC_VER > 1900
#endif // defined(_MSC_VER)

#define VIPL_POINT_DETECTOR_MAJOR_VERSION     5
#define VIPL_POINT_DETECTOR_MINOR_VERSION     0
#define VIPL_POINT_DETECTOR_SUBMINOR_VERSION  0

#include <memory>
#include <vector>

#include "VIPLStruct.h"

/**
 * \brief �����㶨λ����
 */
class VIPLStablePointDetector;

#if _MSC_VER >= 1600
extern template std::shared_ptr<VIPLStablePointDetector>;
extern template std::vector<VIPLPoint>;
#endif

/**
 * \brief �����㶨λ��
 */
class VIPLPointDetector {
public:
	/**
   * \brief ���춨λ������Ҫ���붨λ��ģ��
   * \param model_path ��λ��ģ��
   * \note ��λ��ģ��һ��Ϊ VIPLPointDetector5.0.pts[x].dat��[x]Ϊ��λ�����
   */
  VIPL_API VIPLPointDetector(const char* model_path = nullptr);

	/**
   * \brief ���ض�λ��ģ�ͣ���ж��ǰ����ص�ģ��
   * \param model_path ģ��·��
   */
  VIPL_API void LoadModel(const char* model_path);

	/**
   * \brief �趨�Ƿ����ȶ�ģ�͹���
   * \param is_stable �Ƿ����ȶ�ģʽ����
   * \note 
   */
  VIPL_API void SetStable(bool is_stable);

	/**
   * \brief ���ص�ǰģ��Ԥ��Ķ�λ��ĸ���
   * \return ��λ��ĸ���
   */
  VIPL_API int LandmarkNum() const;

	/**
   * \brief �ڲü��õ������Ͻ��������㶨λ
   * \param src_img �ü��õ�����ͼ�񣬲�ɫ
   * \param landmarks ָ�򳤶�Ϊ��λ������� VIPLPoint ����
   * \return ֻ�ж�λ�ɹ��󷵻���
   */
  VIPL_API bool DetectCroppedLandmarks(const VIPLImageData &src_img, VIPLPoint *landmarks) const;

	/**
   * \brief �ڲü��õ������Ͻ��������㶨λ
   * \param src_img �ü��õ�����ͼ�񣬲�ɫ
   * \param landmarks Ҫ������������������
   * \return ֻ�ж�λ�ɹ��󷵻���
   */
  VIPL_API bool DetectCroppedLandmarks(const VIPLImageData &src_img, std::vector<VIPLPoint> &landmarks) const;

  /**
  * \brief ��ԭͼ�����Ͻ��������㶨λ
  * \param src_img ԭʼͼ�񣬲�ɫ
  * \param face_info ����λ��
   * \param landmarks ָ�򳤶�Ϊ��λ������� VIPLPoint ����
  * \return ֻ�ж�λ�ɹ��󷵻���
  */
  VIPL_API bool DetectLandmarks(const VIPLImageData &src_img, const VIPLFaceInfo &face_info, VIPLPoint *landmarks) const;

  /**
  * \brief ��ԭͼ�����Ͻ��������㶨λ
  * \param src_img ԭʼͼ�񣬲�ɫ
  * \param face_info ����λ��
  * \param landmarks Ҫ������������������
  * \return ֻ�ж�λ�ɹ��󷵻���
  */
  VIPL_API bool DetectLandmarks(const VIPLImageData &src_img, const VIPLFaceInfo &face_info, std::vector<VIPLPoint> &landmarks) const;

private:
	VIPLPointDetector(const VIPLPointDetector &other) = delete;
	const VIPLPointDetector &operator=(const VIPLPointDetector &other) = delete;

private:
  std::shared_ptr<VIPLStablePointDetector> vipl_stable_point_detector_;
};

#endif // VIPL_POINT_DETECTOR_H_